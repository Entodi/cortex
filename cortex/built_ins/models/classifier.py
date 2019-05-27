'''Simple classifier model

'''


from cortex.plugins import (register_plugin, ModelPlugin)
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from .utils import update_encoder_args

from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


def L1_regularization(model, reg = 1e-6):
    # https://github.com/kevinzakka/pytorch-goodies
    l1_loss = 0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + (reg * torch.sum(torch.abs(param)))
    return l1_loss


def L2_regularization(model, reg = 1e-6):
    # https://github.com/kevinzakka/pytorch-goodies
    l2_loss = 0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5 * reg * torch.sum(torch.pow(param, 2)))
    return l2_loss


class SimpleClassifier(ModelPlugin):
    '''Build a simple feed-forward classifier.

    '''
    defaults = dict(
        data=dict(batch_size=128),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(epochs=200, save_on_best='losses.classifier'),
        classifier_args=dict(dropout=0.2))

    def build(self, model_name=None, dim_in: int=None, classifier_args=dict(dim_h=[200, 200]),
         l1_regularization=False, l2_regularization=False):
        '''

        Args:
            dim_in (int): Input size
            classifier_args: Extra arguments for building the classifier

        '''
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.model_name = model_name or 'x'
        dim_l = self.get_dims('labels')
        classifier = FullyConnectedNet(dim_in, dim_out=dim_l, **classifier_args)
        self.nets.classifier = classifier

    def routine(self, inputs, targets,
                criterion=nn.CrossEntropyLoss(reduce=False)):
                #criterion=nn.CrossEntropyLoss(
                #    reduce=False, weight=torch.FloatTensor(
                #        [[[0.34206471, 0.85824345, 0.90909091, 0.89060092]]]).cuda())):
        '''

        Args:
            criterion: Classifier criterion.

        '''
        classifier = self.nets.classifier
        
        outputs = classifier(inputs)
        #print (outputs.size())
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        unlabeled = targets.eq(-1).long()
        losses = criterion(outputs, (1 - unlabeled) * targets)
        labeled = 1. - unlabeled.float()
        loss = (losses * labeled).sum() / labeled.sum()
        if self.l1_regularization:
            l1_loss = L1_regularization(classifier)
            #self.results.l1_loss = l1_loss
            loss = loss + l1_loss
        if self.l2_regularization:
            l2_loss = L2_regularization(classifier)
            #self.results.l2_loss = l2_loss
            loss = loss + l2_loss
       

        bacc = balanced_accuracy_score (targets.cpu(), predicted.cpu())

        if labeled.sum() > 0:
            correct = 100. * (labeled * predicted.eq(
                targets.data).float()).cpu().sum() / labeled.cpu().sum()
            if self.model_name:
                self.results['{}_accuracy'.format(self.model_name)] = correct.item()
                self.results['{}_balanced_accuracy'.format(self.model_name)]  = bacc
                self.results['{}_f1_macro'.format(self.model_name)] = f1_score(targets.cpu(), predicted.cpu(), average='macro')  
            else:
                self.results['accuracy']  = correct
                self.results['balanced_accuracy']  = bacc
            self.losses.classifier = loss
        self.results.perc_labeled = labeled.mean().item()

    def predict(self, inputs):
        classifier = self.nets.classifier

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        return predicted

    def visualize(self, images, inputs, targets):
        predicted = self.predict(inputs)
        self.add_image(images.data, labels=(targets.data, predicted.data),
                       name='gt_pred')


class SimpleAttributeClassifier(SimpleClassifier):
    '''Build a simple feed-forward classifier.

        '''

    defaults = dict(
        data=dict(batch_size=128),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(epochs=200, save_on_best='losses.classifier'))

    def build(self, dim_in: int = None, classifier_args=dict(dim_h=[200, 200])):
        '''

        Args:
            dim_in (int): Input size
            dim_out (int): Output size
            dim_h (:obj:`list` of :obj:`int`): Hidden layer sizes
            classifier_args: Extra arguments for building the classifier

        '''
        dim_a = self.get_dims('attributes')
        classifier = FullyConnectedNet(dim_in, dim_out=dim_a, **classifier_args)
        self.nets.classifier = classifier

    def routine(self, inputs, attributes):
        classifier = self.nets.classifier
        outputs = classifier(inputs, nonlinearity='sigmoid')
        loss = torch.nn.BCELoss()(outputs, attributes)

        predicted = (outputs.data >= 0.5).float()
        correct = 100. * predicted.eq(attributes.data).cpu().sum(0) / attributes.size(0)

        self.losses.classifier = loss
        self.results.accuracy = dict(mean=correct.float().mean().item(),
                                     max=correct.max().item(),
                                     min=correct.min().item())

    def predict(self, inputs):
        classifier = self.nets.classifier
        outputs = classifier(inputs)
        predicted = (F.sigmoid(outputs).data >= 0.5).float()

        return predicted

    def visualize(self, images, inputs):
        self.add_image(images.data, name='gt_pred')


class ImageClassification(SimpleClassifier):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''

    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'))

    def build(self, classifier_type='convnet',
              classifier_args=dict(dropout=0.2), Encoder=None):
        '''Builds a simple image classifier.

        Args:
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout,
            batch_norm, layer_norm, etc.

        '''
        classifier_args = classifier_args or {}

        shape = self.get_dims('x', 'y', 'c')
        dim_l = self.get_dims('labels')

        Encoder_, args = update_encoder_args(
            shape, model_type=classifier_type, encoder_args=classifier_args)
        Encoder = Encoder or Encoder_
        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_l, **args)
        self.nets.classifier = classifier


class ImageAttributeClassification(SimpleAttributeClassifier):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''

    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'))

    def build(self, classifier_type='convnet',
              classifier_args=dict(dropout=0.2), Encoder=None):
        '''Builds a simple image classifier.

        Args:
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout,
            batch_norm, layer_norm, etc.

        '''

        classifier_args = classifier_args or {}

        shape = self.get_dims('x', 'y', 'c')
        dim_a = self.get_dims('attributes')

        Encoder_, args = update_encoder_args(
            shape, model_type=classifier_type, encoder_args=classifier_args)
        Encoder = Encoder or Encoder_

        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_a, **args)
        self.nets.classifier = classifier


register_plugin(ImageClassification)
register_plugin(ImageAttributeClassification)
