""" This baseline is an implementation of the Prototypical Networks introduced 
by J. Snell et al. 2017 (https://arxiv.org/pdf/1703.05175). The network 
initilization at meta-training time can be with:
1. Random weights (Meta-learning league)
2. Pretrained weights (Free-style league)
"""
import os
import random
import pickle
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Any, Tuple, List

from network import ResNet
from helpers_protonet import *
try: 
    import pytorch_metric_learning
except:
    os.system("pip install pytorch-metric-learning")
from self_optimal_transport import SOT
from pytorch_metric_learning import losses, miners
from api import MetaLearner, Learner, Predictor
from torchvision import transforms
import copy
# --------------- MANDATORY ---------------
SEED = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)    
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# -----------------------------------------

TRAIN_AUGMENT = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(127.5,127.5),
    transforms.GaussianBlur(15, sigma=(0.1,0.5)),
    transforms.RandomPerspective()
])

def augment(x):
    return TRAIN_AUGMENT(x)

# def apply_dropout(m):     
#     print(type(m))
#     if type(m) == 'torch.nn.modules.dropout.Dropout':         
#         m.train() 

# def apply_dropout(m):
#     for each_module in m.modules():
#         if each_module.__class__.__name__.startswith('Dropout'):
#             each_module.train()

class MyMetaLearner(MetaLearner):

    def __init__(self, 
                 train_classes: int, 
                 total_classes: int,
                 logger: Any) -> None:
        """ Defines the meta-learning algorithm's parameters. For example, one 
        has to define what would be the meta-learner's architecture. 
        
        Args:
            train_classes (int): Total number of classes that can be seen 
                during meta-training. If the data format during training is 
                'task', then this parameter corresponds to the number of ways, 
                while if the data format is 'batch', this parameter corresponds 
                to the total number of classes across all training datasets.
            total_classes (int): Total number of classes across all training 
                datasets. If the data format during training is 'batch' this 
                parameter is exactly the same as train_classes.
            logger (Logger): Logger that you can use during meta-learning 
                (HIGHLY RECOMMENDED). You can use it after each meta-train or 
                meta-validation iteration as follows: 
                    self.log(data, predictions, loss, meta_train)
                - data (task or batch): It is the data used in the current 
                    iteration.
                - predictions (np.ndarray): Predictions associated to each test 
                    example in the specified data. It can be the raw logits 
                    matrix (the logits are the unnormalized final scores of 
                    your model), a probability matrix, or the predicted labels.
                - loss (float, optional): Loss of the current iteration. 
                    Defaults to None.
                - meta_train (bool, optional): Boolean flag to control if the 
                    current iteration belongs to meta-training. Defaults to 
                    True.
        """
        # Note: the super().__init__() will set the following attributes:
        # - self.train_classes (int)
        # - self.total_classes (int)
        # - self.log (function) See the above description for details
        super().__init__(train_classes, total_classes, logger)
        
        # General data parameters
        self.should_train = True
        self.train_tasks = 1500  #1000
        self.val_tasks = 20
        self.val_after = 25
        
        # General model parameters
        self.meta_batch_size = 1
        self.dev = self.get_device()
        self.opt_fn = torch.optim.Adam
        self.model_args = {
            "num_classes": self.train_classes, 
            "dev": self.dev, 
            "num_blocks": 18, 
            "pretrained": False 
        }
        
        # Meta-learner
        self.lr = 8e-5
        self.meta_learner = ResNet(**self.model_args).to(self.dev)
        self.weights = [p.clone().detach().to(self.dev) for p in 
            self.meta_learner.parameters()]
        for p in self.weights:
            p.requires_grad = True
        self.optimizer = self.opt_fn(self.weights, lr=self.lr)
        
        # Validation-learner
        self.best_score = -float("inf")
        self.best_state = None

    def meta_fit(self, 
                 meta_train_generator: Iterable[Any], 
                 meta_valid_generator: Iterable[Any]) -> Learner:
        """ Uses the generators to tune the meta-learner's parameters. The 
        meta-training generator generates either few-shot learning tasks or 
        batches of images, while the meta-valid generator always generates 
        few-shot learning tasks.
        
        Args:
            meta_train_generator (Iterable[Any]): Function that generates the 
                training data. The generated can be a N-way k-shot task or a 
                batch of images with labels.
            meta_valid_generator (Iterable[Task]): Function that generates the 
                validation data. The generated data always come in form of 
                N-way k-shot tasks.
                
        Returns:
            Learner: Resulting learner ready to be trained and evaluated on new
                unseen tasks.
        """
        if self.should_train:
            self.optimizer.zero_grad()
            for i, task in enumerate(meta_train_generator(self.train_tasks)):
                loss_func = losses.TripletMarginLoss()
                miner = miners.MultiSimilarityMiner()
                self.meta_learner.train()
                
                # Prepare data
                num_ways = task.num_ways
                X_train, y_train, _ = task.support_set
                X_train = augment(X_train)
                X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)
                X_test, y_test, _ = task.query_set
                X_test, y_test = X_test.to(self.dev), y_test.to(self.dev)
                
                # Optimize metalearner
                out, class_loss = self.compute_out_and_loss(X_train, y_train, X_test, 
                    y_test, True, num_ways)

                weights = [p.clone().detach().to(self.dev) for p in self.meta_learner.parameters()]
                feature = self.meta_learner.forward_weights(X_train, weights, embedding=True)
                hard_pairs = miner(feature, y_train)
                contrastive_loss = loss_func(feature, y_train, hard_pairs)
                #print(class_loss, contrastive_loss)
                loss = 10*contrastive_loss + class_loss
                loss.backward()
                if (i + 1) % self.meta_batch_size == 0: 
                    self.optimizer.step()  
                    self.optimizer.zero_grad()
                    
                # Log iteration
                self.log(task, out.detach().cpu().numpy(), loss.item())
                
                if (i + 1) % self.val_after == 0:
                    self.meta_valid(meta_valid_generator)
                    
        if self.best_state is None:
            self.best_state = [p.clone().detach() for p in self.weights]
        
        return MyLearner(self.model_args, self.best_state)
    
    def meta_valid(self, meta_valid_generator: Iterable[Any]) -> None:
        """ Evaluate the current meta-learner with the meta-validation split 
        to select the best model.

        Args:
            meta_valid_generator (Iterable[Task]): Function that generates the 
                validation data. The generated data always come in form of 
                N-way k-shot tasks.
        """
        total_test_images = 0
        correct_predictions = 0
        self.meta_learner.eval()
        for task in meta_valid_generator(self.val_tasks):
            # Prepare data
            num_ways = task.num_ways
            X_train, y_train, _ = task.support_set
            X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)
            X_test, y_test, _ = task.query_set
            X_test = X_test.to(self.dev)
            
            # Evaluate learner
            out, _ = self.compute_out_and_loss(X_train, y_train, X_test, 
                y_test, False, num_ways, True)
            preds = torch.argmax(out, dim=1).cpu().numpy()

            # self.meta_learner.apply(apply_dropout)
            # all_outs = []
            # for i in range(30):
            #     self.meta_learner.apply(apply_dropout)
            #     out, _ = self.compute_out_and_loss(X_train, y_train, X_test, y_test, False, num_ways, True)
            #     print(out)
            #     all_outs.append(out)
            #     preds = torch.argmax(out, dim=1).cpu().numpy()
            # print(' ')
            # all_outs = torch.stack(all_outs, dim=2)
            # out = torch.mean(all_outs, dim=2)

            # Log iteration
            self.log(task, out.cpu().numpy(), meta_train=False)
            
            # Keep track of scores
            total_test_images += len(y_test)
            correct_predictions += np.sum(preds == y_test.numpy())

        # Check if the accuracy is better and store the new best state
        val_acc = correct_predictions / total_test_images
        if val_acc > self.best_score:
            self.best_score = val_acc
            self.best_state = [p.clone().detach() for p in self.weights]

    def get_device(self) -> torch.device:
        """ Get the current device, it can be CPU or GPU.

        Returns:
            torch.device: Available device.
        """
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def compute_out_and_loss(self,
                             X_train: torch.Tensor, 
                             y_train: torch.Tensor, 
                             X_test: torch.Tensor, 
                             y_test: torch.Tensor,
                             training: bool, 
                             num_classes: int,
                             no_loss: bool = False) -> Tuple[torch.Tensor, 
                                                             torch.Tensor]:
        """ Compute the output and loss using the specified data.

        Args:
            X_train (torch.Tensor): Support set images.
            y_train (torch.Tensor): Support set labels.
            X_test (torch.Tensor): Query set images.
            y_test (torch.Tensor): Query set labels.
            training (bool): Boolean flag to control the execution context. If 
                True, keep track of the gradients, otherwise the gradients are 
                ignored.
            num_classes (int): Number of classes to predict.
            no_loss (bool, optional): Boolean flag to control the loss 
                computation. If True, the loss is not computed, otherwise the 
                loss is computed. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and loss.
        """
        # Use torch.no_grad when evaluating
        if training:
            context = self.empty_context
        else:
            context = torch.no_grad

        with context():
            prototypes = process_support_set(self.meta_learner, self.weights, 
                X_train, y_train, num_classes)
            out = process_query_set(self.meta_learner, self.weights, X_test, 
                prototypes)
            if no_loss:
                loss = None
            else:
                loss = self.meta_learner.criterion(out, y_test) / (num_classes 
                    * len(y_test)) 
            
        return out, loss

    @contextlib.contextmanager
    def empty_context(self) -> None:
        """ Defines an empty context to avoid computing unnecessary gradients.
        """
        yield None


class MyLearner(Learner):

    def __init__(self, 
                 model_args: dict = {}, 
                 weights: List[torch.Tensor] = []) -> None:
        """ Defines the learner initialization.
        
        Args:
            model_args (dict, optional): Arguments to initialize the learner. 
                Defaults to {}.
            weights (List[torch.Tensor], optional): Best weights found by the 
                meta-learner. Defaults to [].
        """
        super().__init__()
        self.model_args = model_args
        self.weights = weights

    def fit(self, support_set: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                               int, int]) -> Predictor:
        """ Fit the Learner to the support set of a new unseen task. 
        
        Args:
            support_set (Tuple[Tensor, Tensor, Tensor, int, int]): Support set 
                of a task. The data arrive in the following format (X_train, 
                y_train, original_y_train, n_ways, k_shots). X_train is the 
                tensor of labeled images of shape [n_ways*k_shots x 3 x 128 x 
                128], y_train is the tensor of encoded labels (Long) for each 
                image in X_train with shape of [n_ways*k_shots], 
                original_y_train is the tensor of original labels (Long) for 
                each image in X_train with shape of [n_ways*k_shots], n_ways is
                the number of classes and k_shots the number of examples per 
                class.
                        
        Returns:
            Predictor: The resulting predictor ready to predict unlabelled 
                query image examples from new unseen tasks.
        """

        X_train, y_train, _, n_ways, k = support_set
        X_train_cont, y_train_cont = copy.deepcopy(X_train), copy.deepcopy(y_train)
        X_train, y_train = X_train.to(self.dev), y_train.to(self.dev)

        #X_train, y_train = X_train, y_train
        
        # train_set = torch.utils.data.TensorDataset(X_train_cont, y_train_cont)
        # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=n_ways * k, shuffle=True)
        # optimizer = torch.optim.Adam(self.learner.parameters(), lr=3e-4)

        # miner = miners.MultiSimilarityMiner()
        # loss_func = losses.TripletMarginLoss()
        # train_epochs = 20
        # for epoch in range(0, train_epochs):
        #     train_loss = 0
        #     for i, batch in enumerate(train_dataloader):
                
        #         X_train_cont, y_train_cont = batch
        #         X_train_cont = X_train_cont.to(self.dev)
        #         y_train_cont = y_train_cont.to(self.dev)

        #         optimizer.zero_grad()

        #         weights = [p.clone().detach().to(self.dev) for p in self.learner.parameters()]
        #         feature = self.learner.forward_weights(X_train_cont, weights, embedding=True)

        #         hard_pairs = miner(feature, y_train_cont)
        #         #print(feature)
        #         loss = loss_func(feature, y_train_cont, hard_pairs)
        #         loss.requires_grad = True
        #         print('Loss:',loss)
        #         train_loss += loss.item()
        #         loss.backward()
        #         optimizer.step()

        #     print('epoch [{}/{}], contrastive loss: {:.6f}'.format(epoch+1, train_epochs, train_loss))
        
        with torch.no_grad():
            prototypes = process_support_set(self.learner, self.weights, 
                X_train, y_train, n_ways)
        
        return MyPredictor(self.learner, self.weights, self.dev, prototypes)
        #return MyPredictor(self.learner, self.weights, self.dev, X_train, y_train, n_ways, k)
    
    def save(self, path_to_save: str) -> None:
        """ Saves the learning object associated to the Learner. 
        
        Args:
            path_to_save (str): Path where the learning object will be saved.
        """
        
        if not os.path.isdir(path_to_save):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        with open(f"{path_to_save}/model_args.pickle", "wb+") as f:
            pickle.dump(self.model_args, f)
        with open(f"{path_to_save}/weights.pickle", "wb+") as f:
            pickle.dump(self.weights, f)
 
    def load(self, path_to_load: str) -> None:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in self.save().
        
        Args:
            path_to_load (str): Path where the Learner is saved.
        """
        if not os.path.isdir(path_to_load):
            raise ValueError(("The model directory provided is invalid. Please"
                + " check that its path is valid."))
        
        model_args_file = f"{path_to_load}/model_args.pickle"
        if os.path.isfile(model_args_file):
            with open(model_args_file, "rb") as f:
                self.model_args = pickle.load(f)
            self.dev = self.model_args["dev"]
            self.learner = ResNet(**self.model_args).to(self.dev)
            self.learner.train()
        else:
            raise Exception(f"'{model_args_file}' not found")
        
        weights_file = f"{path_to_load}/weights.pickle"
        if os.path.isfile(weights_file):
            with open(weights_file, "rb") as f:
                self.weights = pickle.load(f)
            for p in self.weights:
                p.requires_grad = True
        else:
            raise Exception(f"'{weights_file}' not found")
        
    
class MyPredictor(Predictor):

    def __init__(self, 
                 model: nn.Module, 
                 weights: List[torch.Tensor],
                 dev: torch.device,
                 prototypes) -> None:
        """Defines the Predictor initialization.

        Args:
            model (nn.Module): Fitted learner.
            weights (List[torch.Tensor]): Best weights for the model.
            dev (torch.device): Device where the data is located.
            prototypes (torch.device): Support prototypes.
        """
        super().__init__()
        self.model = model
        self.weights = weights
        self.dev = dev
        self.prototypes = prototypes
        #self.other = [supp_x, supp_y, n, k]

    def predict(self, query_set: torch.Tensor) -> np.ndarray:
        """ Given a query_set, predicts the probabilities associated to the 
        provided images or the labels to the provided images.
        
        Args:
            query_set (Tensor): Tensor of unlabelled image examples of shape 
                [n_ways*query_size x 3 x 128 x 128].
        
        Returns:
            np.ndarray: It can be:
                - Raw logits matrix (the logits are the unnormalized final 
                    scores of your model). The matrix must be of shape 
                    [n_ways*query_size, n_ways]. 
                - Predicted label probabilities matrix. The matrix must be of 
                    shape [n_ways*query_size, n_ways].
                - Predicted labels. The array must be of shape 
                    [n_ways*query_size].
        """
        # query_set = query_set
        # supp_x, supp_y, n, k = self.other
        # supp_x = supp_x[supp_y.sort()[1]]
        # supp_y_sorted, _ = supp_y.sort()
        # end = supp_x.size(0)
        # x = torch.cat([supp_x, query_set]).to(self.dev)

        # weights = [p.clone().detach().to(self.dev) for p in self.model.parameters()]
        # feature = self.model.forward_weights(x, weights, embedding=True)

        # #x = torch.nn.functional.normalize(feature, dim=1)
        # x  = whiten(x)
        # self.optimal = SOT(distance_metric='cosine')
        # print(x.shape)
        # x = self.optimal(x.to(self.dev), y_support = supp_y_sorted.to(self.dev))

        # with torch.no_grad():
        #     self.prototypes = process_support_set_2(self.model, self.weights, x[:end], supp_y, n)

        # X_test = x[end:].to(self.dev)
        # with torch.no_grad():
        #     out = process_query_set_2(self.model, self.weights, X_test, 
        #         self.prototypes)
        #     probs = F.softmax(out, dim=1).cpu().numpy()


        # self.model.apply(apply_dropout)
        X_test = query_set.to(self.dev)
        # all_outs = []
        # with torch.no_grad():
        #     for i in range(30):

        #         out = process_query_set(self.model, self.weights, X_test, self.prototypes)
        #         all_outs.append(out)
            
        #     all_outs = torch.stack(all_outs, dim=2)
        #     out = torch.mean(all_outs, dim=2)
        #     probs = F.softmax(out, dim=1).cpu().numpy()

        with torch.no_grad():
            out = process_query_set(self.model, self.weights, X_test, 
                self.prototypes)
            probs = F.softmax(out, dim=1).cpu().numpy()
        return probs
