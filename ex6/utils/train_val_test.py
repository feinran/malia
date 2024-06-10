import math

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.notebook import tqdm

from .disc_loss import Identity
from .label import PredictionType, label
from .metric import evaluate
from .visualize import show_instance_segmentation


def unpad(pred, padding_size):
    return pred[padding_size:-padding_size, padding_size:-padding_size]


def calc_accuracy(y_pred, y_true):
    """Calculate the accuracy of the prediction.

    Args:
        y_pred:
            The predicted values
        y_true:
            The true values

    Returns:
         The accuracy

    """
    y_pred = torch.ge(y_pred, 0.5)

    return y_pred.eq(y_true).float().mean()


def save_model_state(epoch, net, optimizer, loss, filename):
    """Save the current state of the model.

    Args:
        epoch:
            The current epoch
        net:
            The model
        optimizer:
            The optimizer
        loss:
            The current loss
        filename:
            The filename to save the model to

    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename)


def get_cropping(in_size, out_size):
    """Compute cropping slices to apply on the input tensor to get the output size.

    Args:
        in_size:
            input size. Assume to be B x C x H x W
        out_size:
            output size. Assume to be B x C x H x W

    Returns:
        A list of slices
    """
    diff_size = np.array(in_size[2:]) - np.array(out_size[2:])  # height and width difference

    # compute the slices to crop the input
    if np.any(diff_size > 0):
        slices = [slice(None)] * 2 + [slice(math.floor(i / 2), - math.ceil(i / 2)) for i in diff_size]
    else:
        slices = [slice(None)] * len(in_size)

    return slices


def train_val_step(model, loss_fn, optimizer, feature, label, activation=None,
                   prediction_type=PredictionType.TWO_CLASS):
    """Train or validation step based on the model mode.

    Args:
        activation: 
        model:
            The model to train or validate. Put the model in train mode or eval mode depending on the use case.
        loss_fn:
            The loss function to use.
        optimizer:
            The optimizer to use.
        feature:
             The input feature.
        label:
            The label.
        prediction_type:
            The prediction type.

    Returns:
        loss_value, outputs
    """
    # speedup version of setting gradients to zero
    for param in model.parameters():
        param.grad = None

    # forward
    logits = model(feature)  # B x C x H x W

    # todo: explain
    shape_dif = np.array(label.shape[-2:]) - np.array(logits.shape[-2:])
    if np.sum(shape_dif) > 0:
        label = label[:, :, shape_dif[0] // 2:-shape_dif[0] // 2, shape_dif[0] // 2:-shape_dif[0] // 2]

    if prediction_type == PredictionType.THREE_CLASS:
        label = torch.squeeze(label, 1)  # label.shape=[N,H,W]

    # loss
    loss_value = loss_fn(input=logits, target=label)  # label.squeeze(0) for three_class

    # training step IFF training mode
    if model.training:
        loss_value.backward()
        optimizer.step()

    # calculate model output
    if activation is not None:
        output = activation(logits)
    else:
        output = logits

    # set outputs
    outputs = {
        'pred': output,
        'logits': logits,
    }

    return loss_value, outputs


def epoch_train_val_routine(net, epochs, learning_rate, train_loader, val_loader, start_epoch=0, optimizer=None,
                            history=None, save_model=False, dtype=torch.FloatTensor, dataset="dsb2018"):
    """Epoch training with validation routine.

    Args:
        dataset:
        net:
            The network to train
        epochs:
             The number of epochs to train
        learning_rate:
             The learning rate
        train_loader:
             The training data loader
        val_loader:
             The validation data loader
        start_epoch:
             The epoch to start with
        optimizer:
             The optimizer to use
        history:
             The history to append to - if None, a new history is created
        save_model:
             True if the model should be saved
        dtype:
             The data type to use

    """
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # set loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = loss_fn.to(device)

    if not list(net.parameters()):
        print("No parameters to optimize")
        return net, history, optimizer

    # set optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # initialize history
    if history is None:
        history = {
            'loss': [],
            'val_loss': [],
            'binary_accuracy': [],
            'val_binary_accuracy': []
        }

    # initialize output variables
    train_acc_loss = []
    epoch = 0

    output_slices = None

    # training and validation loop over epochs
    pbar = tqdm(total=epochs)
    for epoch in range(0, epochs):
        # train routine
        train_acc_accuracy, train_acc_loss, output_slices = simple_train_routine(net, train_loader, loss_fn, optimizer,
                                                                                 device, output_slices, dtype=dtype)

        # update history for epoch
        history['loss'].append(np.mean(train_acc_loss))
        history['binary_accuracy'].append(np.mean(train_acc_accuracy))

        # validation routine
        val_acc_accuracy, val_acc_loss = simple_val_routine(net, val_loader, loss_fn, optimizer, device, output_slices,
                                                            dtype=torch.FloatTensor)

        # update history for epoch
        history['val_loss'].append(np.mean(val_acc_loss))
        history['val_binary_accuracy'].append(np.mean(val_acc_accuracy))

        # show progress
        print(
            f'Epoch {epoch + start_epoch + 1}, train-loss: {np.mean(train_acc_loss):.4f} - train_accuracy:{np.mean(train_acc_accuracy):.4f}' +
            f' - val_loss:{np.mean(val_acc_loss):.4f} -val_accuracy:{np.mean(val_acc_accuracy):.4f}'
        )

        # update progress bar
        pbar.update(1)

    if save_model:
        save_model_state(epoch, net, optimizer, train_acc_loss, "model_{}_{}.pth".format(dataset, epoch))

    print('Finished Training')

    return net, history, optimizer


def step_train_val_routine(net, train_loader, val_loader, training_steps, optimizer, loss_fn, writer, prediction_type,
                           device, activation=None, save_model=False, dtype=torch.FloatTensor, dataset="dsb2018"):
    """Step based training with validation routine.

    Args:
        dataset:
        net:
             The network to train
        train_loader:
             The training data loader
        val_loader:
             The validation data loader
        training_steps:
             The number of training steps
        optimizer:
             The optimizer to use
        activation: 
            The activation function to use after the forward pass
        loss_fn: 
            The loss function to use
        writer:
             The tensorboard writer
        prediction_type:
             The prediction type
        device: 
            The device to use
        save_model:
             The flag to save the model
        dtype:
             The data type

    """
    with tqdm(total=training_steps) as pbar:
        step = 0

        # step based training loop
        while step < training_steps:
            # reset data loader to get random augmentations
            np.random.seed()
            tmp_loader = iter(train_loader)

            # iterate over training data
            for img, label in tmp_loader:
                # prepare data
                label = label.type(dtype)
                label = label.to(device)
                img = img.to(device)

                # train step
                loss_value, pred = train_val_step(net, loss_fn, optimizer, img, label, activation=activation,
                                                  prediction_type=prediction_type)

                # output to tensorboard and update progress bar
                writer.add_scalar('loss', loss_value.cpu().detach().numpy(), step)
                step += 1
                pbar.update(1)

                # validation step
                if step % 100 == 0:
                    # ensure that the network is in evaluation mode
                    net.eval()

                    tmp_val_loader = iter(val_loader)
                    acc_loss = []

                    # validation loop
                    for img, label in tmp_val_loader:
                        # prepare data
                        label = label.type(dtype)
                        label = label.to(device)
                        img = img.to(device)

                        # validation step without training
                        loss_value, _ = train_val_step(net, loss_fn, optimizer, img, label, activation=activation,
                                                       prediction_type=prediction_type)

                        # append loss
                        acc_loss.append(loss_value.cpu().detach().numpy())

                    # output mean validation accuracy to tensorboard
                    writer.add_scalar('val_loss', np.mean(acc_loss), step)

                    # ensure that the network back in training mode
                    net.train()

    if save_model:
        save_model_state(step, net, optimizer, acc_loss, "model_{}_{}.pth".format(dataset, step))


def simple_val_routine(net, val_loader, loss_fn, optimizer, device, output_slices, dtype=torch.FloatTensor):
    """A simple validation routine for a given model.

    Args:
        net:
             The model to train
        val_loader:
             The validation data loader
        loss_fn:
             The loss function
        optimizer:
             The optimizer
        device:
             The device to use
        output_slices:
             The output slices to use
        dtype:
             The data type to use

    Returns:
        Validation accuracy and validation loss

    """
    tmp_val_loader = iter(val_loader)
    # initialize loss and accuracy
    val_acc_loss = []
    val_acc_accuracy = []
    # set model to evaluation mode
    net.eval()
    # validation loop
    for feature, label in tmp_val_loader:
        # prepare the label
        label = label[output_slices]
        label = label.type(dtype)
        label = label.to(device)

        # prepare the feature
        feature = feature.to(device)

        # training step
        loss_value, outputs = train_val_step(net, loss_fn, optimizer, feature, label)

        # update loss and accuracy
        val_acc_loss.append(loss_value.cpu().detach().numpy())
        accuracy = calc_accuracy(outputs['pred'], label)
        val_acc_accuracy.append(float(accuracy.cpu().detach().numpy()))
    return val_acc_accuracy, val_acc_loss


def simple_train_routine(net, train_loader, loss_fn, optimizer, device, output_slices, dtype=torch.FloatTensor):
    """A simple training routine for a given network, data loader, loss function and optimizer.

    Args:
        net:
            The network to train
        train_loader:
            The data loader for the training data
        loss_fn:
            The loss function to use
        optimizer:
            The optimizer to use
        device:
            The device to use
        output_slices:
            The slices to use for the output
        dtype:
            The data type to use

    Returns:
        Training accuracy, training loss and output slices
    """
    np.random.seed()  # reset data loader to get random augmentations
    tmp_loader = iter(train_loader)
    # initialize loss and accuracy
    train_acc_loss = []
    train_acc_accuracy = []
    # set model to training mode
    net.train()
    # training loop
    for feature, label in tmp_loader:
        # prepare the label
        if output_slices is None:
            rand_tensor = torch.rand(feature.shape).to(device)
            net_output_shape = net(rand_tensor).shape
            output_slices = get_cropping(label.shape, net_output_shape)
        label = label[output_slices]
        label = label.type(dtype)
        label = label.to(device)

        # prepare the feature
        feature = feature.to(device)

        # training step
        loss_value, outputs = train_val_step(net, loss_fn, optimizer, feature, label)

        # save loss and accuracy
        train_acc_loss.append(loss_value.cpu().detach().numpy())
        accuracy = calc_accuracy(outputs['pred'], label)
        train_acc_accuracy.append(float(accuracy.cpu().detach().numpy()))

    return train_acc_accuracy, train_acc_loss, output_slices


def simple_test(net, test_loader, activation, device):
    """Predicts the output of the network for the given test_loader.

    Args:
        activation: 
        device: 
        net:
            The network to test.
        test_loader:
            The test loader.

    Returns:
        Images, labels, predictions, mean accuracy.
    """
    # set model to evaluation mode
    net.eval()

    # initialize output and accuracy
    predictions = []
    imgs = []
    labels = []
    acc_accuracy = []
    output_slices = None

    # test loop
    for image, label in test_loader:

        # prepare the label
        if output_slices is None:
            rand_tensor = torch.rand(image.shape).to(device)
            net_output_shape = net(rand_tensor).shape
            output_slices = get_cropping(label.shape, net_output_shape)
        image = image.to(device)
        label = label[output_slices]
        label = label.to(device)

        # predict
        pred = net(image)
        pred = activation(pred)

        # calculate accuracy
        accuracy = calc_accuracy(pred, label)
        acc_accuracy.append(float(accuracy.cpu().detach().numpy()))

        # to cpu and squeeze to remove batch dimension
        np_img = np.squeeze(image.cpu().detach().numpy(), 0)
        np_label = np.squeeze(label.cpu().detach().numpy(), 0)
        pred = np.squeeze(pred.cpu().detach().numpy(), 0)

        # store prediction and image
        predictions.append(pred)
        imgs.append(np_img)
        labels.append(np_label)

    predictions = np.stack(predictions, axis=0)
    imgs = np.stack(imgs, axis=0)
    labels = np.stack(labels, axis=0)

    return imgs, labels, predictions, float(np.mean(acc_accuracy))


def test_evaluation_routine(net, test_loader, device, prediction_type, fg_thresh, seed_thresh, padding_size, data_test):
    """Perform post-processing on the network output and evaluate and visualize the results.

    Args:
        data_test:
        net:
            The network to test.
        test_loader:
            The test loader.
        device:
            The device to use.
        prediction_type:
            The prediction type.
        fg_thresh:
            The foreground threshold for post-processing.
        seed_thresh:
            The seed threshold for post-processing.
        padding_size:
            The padding size.

    """

    # catch identity network
    if isinstance(net[0], Identity):
        return

    avg = 0.0
    input_size = None
    output_size = None
    pad_h = None
    pad_w = None
    idx = 0

    net.eval()
    for idx, (image, gt_labels) in enumerate(test_loader):
        # get network output size
        if output_size is None:
            input_size = image.shape  # [-2:]
            rand_tensor = torch.rand(image.shape).to(device)
            output_size = net(rand_tensor).shape
            pad_h = (input_size[2] - output_size[2]) // 2  # assume size with shape B x C x H x W
            pad_w = (input_size[3] - output_size[3]) // 2  # assume size with shape B x C x H x W

        # predict the image
        image = image.to(device)
        if pad_h == 0 and pad_w == 0:
            pred = net(image)
        else:
            pred = tile_and_stitch_predict(net, image, input_size, output_size, pad_h, pad_w)

        # prepare image and method ground truth
        image = np.squeeze(image.cpu())
        gt_labels = np.squeeze(gt_labels)

        # prepare network prediction
        pred = np.squeeze(pred.cpu().detach().numpy(), 0)
        if padding_size and padding_size > 0:
            pred = unpad(np.transpose(pred, (1, 2, 0)), padding_size)
            pred = np.transpose(pred, (2, 0, 1))

        # affinities method has two label channels
        if prediction_type == PredictionType.AFFINITIES:
            gt_labels = gt_labels[0] + gt_labels[1]

        # get instance segmentation
        instance_segmentation, surface = label(pred, prediction_type, fg_thresh=fg_thresh, seed_thresh=seed_thresh)

        # evaluate instance segmentation
        ap, precision, recall, tp, fp, fn = evaluate(instance_segmentation, data_test.get_filename(idx))

        # sum up average precision
        avg += ap

        # print results
        print(np.min(surface), np.max(surface))
        print("average precision: {}, precision: {}, recall: {}".format(ap, precision, recall))
        print("true positives: {}, false positives: {}, false negatives: {}".format(tp, fp, fn))

        # prepare surface for metric learning case
        if prediction_type == PredictionType.METRIC_LEARNING:
            surface = surface + np.abs(np.min(surface, axis=(1, 2)))[:, np.newaxis, np.newaxis]
            surface /= np.max(surface, axis=(1, 2))[:, np.newaxis, np.newaxis]
            surface = np.transpose(surface, (1, 2, 0))

        # retrieve original image and ground truth shape
        instance_segmentation = instance_segmentation.astype(np.uint8)
        image = unpad(image, (image.shape[-1] - instance_segmentation.shape[-1]) // 2)
        gt_labels = unpad(gt_labels, (gt_labels.shape[-1] - instance_segmentation.shape[-1]) // 2)

        # visualize results
        show_instance_segmentation(image, gt_labels, surface, instance_segmentation)

    # calculate average precision and report
    avg /= (idx + 1)
    print("average precision on test set: {}".format(avg))


def tile_and_stitch_predict(net, image, input_size, output_size, pad_h, pad_w):
    """Perform prediction on the image by tiling and stitching.

    Args:
        net: 
        image:
            The image to predict on.
        input_size:
            The input size of the network.
        output_size:
            The output size of the network.
        pad_h:
            The padding size of the network in height.
        pad_w:
            The padding size of the network in width.

    Returns:
        The prediction of the network on the image.

    """
    B, C, H, W = image.shape
    Bi, Ci, Hi, Wi = input_size
    Bo, Co, Ho, Wo = output_size

    hfactor = math.ceil((Hi + pad_h) / Hi)
    wfactor = math.ceil((Wi + pad_w) / Wi)

    bottom_h_pad = hfactor * Hi - (Hi + pad_h + (Hi-Ho))
    bottom_w_pad = wfactor * Wi - (Wi + pad_w + (Wi-Wo))
    image_padded = F.pad(image, (pad_w, bottom_w_pad, pad_h, bottom_h_pad))
    patched_pred = torch.zeros(B, Co, hfactor*Ho, wfactor*Wo)
    for h in range(0, H, Ho):
        for w in range(0, W, Wo):
            image_tmp = image_padded[:, :, h:h + Hi, w:w + Wi]
            pred = net(image_tmp)
            patched_pred[:, :, h:h + pred.shape[2], w:w + pred.shape[3]] = pred

    pred = patched_pred[:, :, 0:Hi, 0:Wi]

    return pred
