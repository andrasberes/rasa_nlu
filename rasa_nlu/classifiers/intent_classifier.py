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
        return vec.view(1, -1)
    
    def make_target(self, label, indexed_labels):
        return torch.LongTensor([indexed_labels[label]])

    def train(self, training_data, config=None, **kwargs):
        
        labels = [e.get("intent") for e in training_data.intent_examples]

        text_features = np.stack([example.get("text_features") for example in training_data.intent_examples])
        text_features_list = [example.get("text_features") for example in training_data.intent_examples]

        indexed_features = self.feature2ix(text_features)
        indexed_features_list = self.feature2ix(text_features_list)

        self.linear = nn.Linear(len(indexed_features), len(set(labels)))
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        
        real_training_data = [(text_features_list[i],labels[i]) for i in range(len(labels))]

        if len(set(labels)) < 2:
            logger.warn("Can not train an intent classifier. Need at least 2 different classes. " +
                        "Skipping training of intent classifier.")
        else:
            for epoch in range(40):
                print("Training neural network...epoch {}".format(epoch))
                for instance, label in real_training_data:
                    self.zero_grad()	#clear gradient each turn before calculating  
                    bow_vec = autograd.Variable(self.make_vector(instance, indexed_features))
                    target = autograd.Variable(self.make_target(label, self.indexed_labels))
                    
                    # Step 3. Run our forward pass.
                    log_probs = self.forward(bow_vec)
                    
                    # Step 4. Compute the loss, gradients, and update the parameters by
                    # calling optimizer.step()
                    loss = self.loss_function(log_probs, target)
                    loss.backward()
                    self.optimizer.step()
                    
    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle
        import os, io
        classifier_file = os.path.join(model_dir, "intent_classifier.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)
        return {
            "intent_classifier_pytorch": "intent_classifier.pkl"
        }
