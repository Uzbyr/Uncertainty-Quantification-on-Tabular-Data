import math
from copy import deepcopy
from typing import Any, Optional
from typing import Optional

import scipy.special
from sklearn.impute import SimpleImputer
import sklearn.metrics
import pandas as pd
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch import Tensor
from tabm import TabM
from sklearn.base import BaseEstimator
from mapie.utils import ClassifierMixin


class TabMClassifer(BaseEstimator, ClassifierMixin):
    def __init__(self, n_num_features, cat_cardinalities, n_classes, num_embeddings, train_size, device) -> None:
        self.cat_cardinalities = cat_cardinalities
        self.n_num_features = n_num_features
        self.num_embeddings = num_embeddings
        self.n_classes = n_classes
        self.train_size = train_size
        self.model = TabM.make(n_num_features=n_num_features, d_out=n_classes,
                               cat_cardinalities=cat_cardinalities,
                               num_embeddings=num_embeddings).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=3e-4)
        self.device = device
        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else None
        )
        # Changing False to True can speed up training
        # of large enough models on compatible hardware.
        self.amp_enabled = False and self.amp_dtype is not None
        # self.grad_scaler = torch.cuda.amp.GradScaler() if self.amp_dtype is torch.float16 else None
        self.grad_scaler = torch.amp.GradScaler('cuda') if self.amp_dtype is torch.float16 else None
        self.share_training_batches = True
        self.base_loss_fn = nn.functional.cross_entropy
        self.patience = 16
        self.n_epochs = 1_000_000_000
        self.batch_size = 256
        self.gradient_clipping_norm: Optional[float] = 1.0
        self.best_checkpoint = self.__make_checkpoint(-1, -math.inf)


    def fit(self, X_trainval, y_trainval):
        self.classes_ = np.unique(y_trainval)
        self._is_fitted = True

        X_train = X_trainval[:self.train_size]
        X_val = X_trainval[self.train_size:]

        self._imputer = SimpleImputer().fit(X_train)
        X_train = self._imputer.transform(X_train)
        X_val = self._imputer.transform(X_val)

        return self

        y_train = y_trainval[:self.train_size]
        y_val = y_trainval[self.train_size:]
        X_train = torch.as_tensor(X_train.to_numpy(), device=self.device)
        X_val = torch.as_tensor(X_val.to_numpy(), device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        y_val = torch.as_tensor(y_val, dtype=torch.long, device=self.device)
        # epoch_size = math.ceil(train_size / self.batch_size)

        remaining_patience = self.patience

        for epoch in range(self.n_epochs):
            batches = (
                # Create one standard batch sequence.
                torch.randperm(train_size, device=self.device).split(self.batch_size)
                if self.share_training_batches
                # Create k independent batch sequences.
                else (
                    torch.rand((train_size, self.model.backbone.k), device=self.device)
                    .argsort(dim=0)
                    .split(self.batch_size, dim=0)
                )
            )

            for batch_idx in batches:
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.__loss_fn(self.__apply_model(X_train, batch_idx), y_train[batch_idx])
                if self.gradient_clipping_norm is not None:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping_norm
                    )
                if self.grad_scaler is None:
                    loss.backward()
                    self.optimizer.step()
                else:
                    self.grad_scaler.scale(loss).backward()  # type: ignore
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

            metrics = self.__evaluate(X_val, y_val)
            val_score_improved = metrics > self.best_checkpoint['metrics']

            # print(
            #     f'{"*" if val_score_improved else " "}'
            #     f' [epoch] {epoch:<3}'
            #     f' [val] {metrics["val"]:.3f}'
            #     f' [test] {metrics["test"]:.3f}'
            # )

            if val_score_improved:
                self.best_checkpoint = self.__make_checkpoint(epoch, metrics)
                remaining_patience = self.patience
            else:
                remaining_patience -= 1

            if remaining_patience < 0:
                break

        self._is_fitted = True
        return self


    def predict_proba(self, X):
        self.model.load_state_dict(self.best_checkpoint['model'])
        self.model.eval()
        X = self._imputer.transform(X)
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits = self.model(xb)
            probs = torch.softmax(logits.mean(dim=1), dim=1)
            return probs.cpu().numpy()


    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


    def __apply_model(self, X: Tensor, idx: Tensor) -> Tensor:
        with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
            output = self.model(X[idx])
        return output.float()


    def __loss_fn(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions. Each of them must be trained separately.

        # Regression:     (batch_size, k)            -> (batch_size * k,)
        # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
        y_pred = y_pred.flatten(0, 1)

        if self.share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            y_true = y_true.repeat_interleave(self.model.backbone.k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)

        return self.base_loss_fn(y_pred, y_true)


    @torch.inference_mode()
    def __evaluate(self, X, y) -> float:
        self.model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 8096
        y_pred: np.ndarray = (
            torch.cat(
                [
                    self.__apply_model(X, idx)
                    for idx in torch.arange(len(y), device=self.device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(1)

        y_true = y.cpu().numpy()
        score = sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        return float(score)  # The higher -- the better.

    def __make_checkpoint(self, epoch: int, metrics: float) -> dict[str, Any]:
        return deepcopy(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
            }
        )

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
