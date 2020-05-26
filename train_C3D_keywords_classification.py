import os
from utils.utils import *
import torch.optim as optim
from utils.opts_C3D_videos import *
from torch.utils.data import DataLoader
from model.Model_S3D_videos import Model
from model.transformer.Optim import ScheduledOptim
from utils.dataloader_howto100 import HowTo100MDataset, HowTo100_collate_fn


def train(loader, model, optimizer, opt):
    model.train()

    for epoch in range(opt['epochs']):
        for (video_input, keywords, batch_lens, video_id) in loader:
            torch.cuda.synchronize()

            # Convert the textual input to numeric labels
            key_gts, key_mask = convert_caption_labels(keywords, loader.dataset.get_vocab(), opt['max_length'])

            video_input = video_input.cuda()
            key_gts = torch.tensor(key_gts).cuda().long()
            # key_mask = key_mask.cuda()
            key_pos = pos_emb_generation(key_gts)

            optimizer.zero_grad()

            key_probs = model(video_input, batch_lens, key_gts, key_pos)

            key_loss, key_n_correct = cal_performance(key_probs, key_gts[:, 1:], smoothing=True)
            loss = key_loss

            #show_prediction(key_probs, key_gts[:, :-1], loader.dataset.get_ix_to_word())

            loss.backward()
            optimizer.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)

            # update parameters
            key_train_loss = key_loss.item()
            torch.cuda.synchronize()

            non_pad_mask = key_gts[:, 1:].ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            print('(epoch %d), cap_train_loss = %.6f, current step = %d, current lr = %.3E, cap_acc = %.3f,'
                  % (epoch, key_train_loss/n_word, optimizer.n_current_steps,
                     optimizer._optimizer.param_groups[0]['lr'], key_n_correct/n_word))

        if epoch % opt['save_checkpoint_every'] == 0 and epoch != 0:
            model_path = os.path.join(opt['checkpoint_path'], 'Model_c3d_2tsfm_%d.pth' % epoch)
            model_info_path = os.path.join(opt['checkpoint_path'], 'model_score3.txt')
            torch.save(model.state_dict(), model_path)

            print('model saved to %s' % model_path)
            with open(model_info_path, 'a') as f:
                f.write('model_%d, cap_loss: %.6f' % (epoch, key_train_loss))


def main(opt):
    # mode = train|val
    dataset = HowTo100MDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True, collate_fn=HowTo100_collate_fn)

    model = Model(
        dataset.get_vocab_size(),
        cap_max_seq=opt['max_length'],
        tgt_emb_prj_weight_sharing=True,
        d_k=opt['dim_head'],
        d_v=opt['dim_head'],
        d_model=opt['dim_model'],
        d_word_vec=opt['dim_word'],
        d_inner=opt['dim_inner'],
        n_layers=opt['num_layer'],
        n_head=opt['num_head'],
        dropout=0.1,
        c3d_path=opt['c3d_path'])

    model = nn.DataParallel(model).cuda()

    # if opt['load_checkpoint']:
    #     state_dict = torch.load(opt['load_checkpoint'])
    #     model.load_state_dict(state_dict)

    optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09), 512, opt['warm_up_steps'])

    train(dataloader, model, optimizer, opt)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = parse_opt()
    opt = vars(opt)
    main(opt)
