from tqdm import tqdm
import torch
from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import configs
from ..utils import dist


class ClassificationTrainer(BaseTrainer):
    def validate(self):
        self.model.eval()
        val_criterion = self.criterion  # torch.nn.CrossEntropyLoss()

        val_loss = DistributedMetric('val_loss')
        val_top1 = DistributedMetric('val_top1')

        with torch.no_grad():
            with tqdm(total=len(self.data_loader['val']),
                      desc='Validate',
                      disable=dist.rank() > 0 or configs.ray_tune) as t:
                for images, labels in self.data_loader['val']:
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = self.model(images)
                    loss = val_criterion(output, labels)
                    val_loss.update(loss, images.shape[0])
                    acc1 = accuracy(output, labels, topk=(1,))[0]
                    val_top1.update(acc1.item(), images.shape[0])

                    t.set_postfix({
                        'loss': val_loss.avg.item(),
                        'top1': val_top1.avg.item(),
                        'batch_size': images.shape[0],
                        'img_size': images.shape[2],
                    })
                    t.update()
        return {
            'val/top1': val_top1.avg.item(),
            'val/loss': val_loss.avg.item(),
        }

    def train_one_epoch(self, epoch):
        import time
        self.model.train()
        self.data_loader['train'].sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1')
        
        # Timing metrics
        iteration_times = []
        batch_processing_times = []
        epoch_start_time = time.time()

        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                iter_start_time = time.time()
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()

                output = self.model(images)
                loss = self.criterion(output, labels)
                # backward and update
                loss.backward()

                # partial update config
                if configs.backward_config.enable_backward_config:
                    from core.utils.partial_backward import apply_backward_config
                    apply_backward_config(self.model, configs.backward_config)

                if hasattr(self.optimizer, 'pre_step'):  # for SGDScale optimizer
                    self.optimizer.pre_step(self.model)

                self.optimizer.step()

                if hasattr(self.optimizer, 'post_step'):  # for SGDScaleInt optimizer
                    self.optimizer.post_step(self.model)

                # after one step
                train_loss.update(loss, images.shape[0])
                acc1 = accuracy(output, labels, topk=(1,))[0]
                train_top1.update(acc1.item(), images.shape[0])
                
                # Record timing
                iter_time = time.time() - iter_start_time
                iteration_times.append(iter_time)
                samples_per_sec = images.shape[0] / iter_time
                batch_processing_times.append(samples_per_sec)

                t.set_postfix({
                    'loss': train_loss.avg.item(),
                    'top1': train_top1.avg.item(),
                    'batch_size': images.shape[0],
                    'img_size': images.shape[2],
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'samples/s': f'{samples_per_sec:.1f}'
                })
                t.update()

                # after step (NOTICE that lr changes every step instead of epoch)
                self.lr_scheduler.step()

        # Calculate timing statistics
        avg_iter_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0
        avg_throughput = sum(batch_processing_times) / len(batch_processing_times) if batch_processing_times else 0
        total_samples = len(self.data_loader['train']) * self.data_loader['train'].batch_size
        
        return {
            'train/top1': train_top1.avg.item(),
            'train/loss': train_loss.avg.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
            'train/avg_iter_time': avg_iter_time,
            'train/throughput_samples_per_sec': avg_throughput,
            'train/total_samples': total_samples
        }
