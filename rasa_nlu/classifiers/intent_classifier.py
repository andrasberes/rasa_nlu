from future.utils import PY3
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.components import Component
import numpy as np
from builtins import zip
import os
import io
import cloudpickle

class PytorchIntentClassifier(nn.Module,Component):

    name = "intent_classifier_pytorch"
  
    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]	

    def __init__(self, training_data=None):
        super(PytorchIntentClassifier, self).__init__()
        
        self.indexed_labels = {"greet": 0, "affirm": 1, "restaurant_search": 2, "goodbye": 3}
    
    def feature2ix(self, text):   
        #Indexed_words maps each feature to a unique integer
        indexed_f = {}
        for i in text:
            for f in i:
                if f not in indexed_f:
                    indexed_f[f] = len(indexed_f)
        return indexed_f

    def forward(self, fea):
        return F.log_softmax(self.linear(fea), dim=1)

    def make_vector(self, feature, indexed_fe):
        vec = torch.zeros(len(indexed_fe))
        for sub_fea in feature:
            vec[indexed_fe[sub_fea]] += 1
        return vec.view(1, -1)      #reshape 
    
    def make_target(self, label, indexed_labels):
        return torch.LongTensor([indexed_labels[label]])

    def train(self, training_data, config=None, **kwargs):
        
        print("Training neural network...")
        labels = [e.get("intent") for e in training_data.intent_examples]
        text_features = np.stack([example.get("text_features") for example in training_data.intent_examples])
        text_features_list = [example.get("text_features") for example in training_data.intent_examples]

        self.indexed_features = self.feature2ix(text_features)
        indexed_features_list = self.feature2ix(text_features_list)

        self.linear = nn.Linear(len(self.indexed_features), len(set(labels)))
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        
        real_training_data = [(text_features_list[i],labels[i]) for i in range(len(labels))]

        if len(set(labels)) < 2:
            logger.warn("Can not train an intent classifier. Need at least 2 different classes. " +
                        "Skipping training of intent classifier.")
        else:
            for epoch in range(40):
                for instance, label in real_training_data:
                    self.zero_grad()	#clear gradient each turn before calculating  
                    bow_vec = autograd.Variable(self.make_vector(instance, self.indexed_features))
                    target = autograd.Variable(self.make_target(label, self.indexed_labels))

                    # Step 3. Run our forward pass.
                    log_probs = self.forward(bow_vec)
                    
                    # Step 4. Compute the loss, gradients, and update the parameters by
                    # calling optimizer.step()
                    loss = self.loss_function(log_probs, target)
                    loss.backward()
                    self.optimizer.step()

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> PytorchIntentClassifier

        if model_dir and model_metadata.get("intent_classifier_pytorch"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_classifier_pytorch"))
            with io.open(classifier_file, 'rb') as f:  # pragma: no test
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return PytorchIntentClassifier()
                    
    #Given a bow vector of an input text, predict most probable label. Returns only the most likely label.
    '''
    def predict(self, X):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """
        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its probability"""

        """Given a bow vector of an input text, predict the intent label. Returns probabilities for all labels.
        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""
        pred_result = self.clf.predict_proba(X)
        # sort the probabilities retrieving the indices of the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]
    '''

                    
    def process(self, message, **kwargs):
    # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""

        X = (message.get("text_features"))
        pytorch_vec = autograd.Variable(self.make_vector(X, self.indexed_features)) #itt az X csak tipp
        log_probs = self.forward(pytorch_vec)
        print("ITT A KIIRAS: log_probs: {}".format(log_probs))
        '''
        intent_ids, probabilities = self.predict(X)
        intent_examples = [e.get("intent") for e in training_data.intent_examples]
        intents = set(intent_examples)
        print("ITTVAN1 intent_labels {}".format(intent_labels))
        #intents = self.transform_labels_num2str(intent_ids)
        
        # `predict` returns a matrix as it is supposed
        # to work for multiple examples as well, hence we need to flatten
        intents, probabilities = intents.flatten(), probabilities.flatten()

        if intents.size > 0 and probabilities.size > 0:
            INTENT_RANKING_LENGTH = 10
            ranking = list(zip(list(intents), list(probabilities)))[:INTENT_RANKING_LENGTH]
            intent = {"name": intents[0], "confidence": probabilities[0]}
            intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
        else:
            intent = {"name": None, "confidence": 0.0}
            intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
        '''

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        
        classifier_file = os.path.join(model_dir, "intent_classifier.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)
        return {
            "intent_classifier_pytorch": "intent_classifier.pkl"
        }
