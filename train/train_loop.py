from tqdm import tqdm

class TrainNetwork:
    def __init__(self, args, writer, model, diffusion_train, diffusion_eval, timestep_respacing_eval,
                 start_infill_epoch, max_infill_ratio, mask_prob, train_dataloader, test_dataloader, logdir, logger, device='cpu'):
        self.model = model
        self.train_dataloader = train_dataloader

        # 
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.train_dataloader) + 1

    def run_loop(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in tqdm(self.train_dataloader):
                for k in batch.keys():
                    batch[k] = batch[k].to(self.device)



                ### Train loop
                train_losses = self.run_step(batch)

                if self.step % self.log_interval == 0 and self.step > 0:
                    for key in train_losses.keys():
                        self.writer.add_scalar('train/{}'.format(key), train_losses[key].item(), self.step)
                        print_str = '[Step {:d}/ Epoch {:d}] [train]  {}: {:.10f}'. format(self.step, epoch, key, train_losses[key].item())
                        self.logger.info(print_str)
                        print(print_str)

                if self.step % self.log_interval == 0 and self.step > 0:
                    self.model.eval()
                    for test_step, test_batch in tqdm(enumerate(self.test_dataloader)):
                        for key in test_batch.keys():
                            test_batch[key] = test_batch[key].to(self.device)
                        shape = list(test_batch['motion_repr_clean'][:, :, 0:traj_feat_dim].shape)
                        eval_losses, val_output = self.diffusion_eval.eval_losses(model=self.model, batch=test_batch,
                                                                                  shape=shape, progress=False,
                                                                                  clip_denoised=False, cur_epoch=epoch,
                                                                                  timestep_respacing=self.timestep_respacing_eval,
                                                                                  smplx_model=self.smplx_neutral)
                        for key in eval_losses.keys():
                            if test_step == 0:
                                eval_losses[key] = eval_losses[key].detach().clone()
                            if test_step > 0:
                                eval_losses[key] += eval_losses[key].detach().clone()

                    for key in eval_losses.keys():
                        eval_losses[key] = eval_losses[key] / (test_step + 1)
                        self.writer.add_scalar('eval/{}'.format(key), eval_losses[key].item(), self.step)
                        print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'.format(self.step, epoch, key, eval_losses[key].item())
                        self.logger.info(print_str)
                        print(print_str)

                    self.model.train()

                if self.step % self.save_interval == 0 and self.step > 0:
                    self.save()

                self.step += 1


    
    def run_step(self, batch):
        losses = self.forward_backward(batch)
        self.mp_trainer.optimize(self.opt)
        return losses

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        t, weights = self.schedule_sampler.sample(batch['motion_repr_clean'].shape[0], dist_util.dev())
        losses = self.diffusion_train.training_losses(model=self.model, batch=batch, t=t, noise=None,
                                                      traj_feat_dim=self.train_dataloader.dataset.traj_feat_dim,
                                                      smplx_model=self.smplx_neutral)
        loss = (losses["loss"] * weights).mean()
        self.mp_trainer.backward(loss)
        return losses