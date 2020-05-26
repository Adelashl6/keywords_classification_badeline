from utils.utils import *
from utils.opts_C3D_videos import *
from torch.utils.data import DataLoader
from model.Model_S3D_videos import Model
from utils.dataloader_howto100 import HowTo100MDataset, HowTo100_collate_fn


def test(loader, model, opt, vocab):

    for batch_id, (video_input, captions, batch_lens, video_id) in enumerate(loader):

        # Convert the textual input to numeric labels
        cap_gts, cap_mask = convert_caption_labels(captions, loader.dataset.get_vocab(), opt['max_length'])

        video_input = video_input.cuda()
        cap_gts = torch.tensor(cap_gts).cuda().long()
        # cap_mask = cap_mask.cuda()
        cap_pos = pos_emb_generation(cap_gts)
        with torch.no_grad():
            cap_probs = model(video_input, batch_lens, cap_gts, cap_pos)
        test_show_prediction(cap_probs, cap_gts[:, :-1], vocab)


def main(opt):
    dataset = HowTo100MDataset(opt, 'val')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False, collate_fn=HowTo100_collate_fn)

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

    model = nn.DataParallel(model)
    state_dict = torch.load(opt['load_checkpoint'])
    model.load_state_dict(state_dict)

    model.eval().cuda()
    test(dataloader, model, opt, dataloader.dataset.get_ix_to_word())


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    opt['batch_size'] = 1
    main(opt)

