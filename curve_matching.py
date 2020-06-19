import cv2
import os
import math
import numpy as np
import caffe
import json
import faiss
import xlsxwriter


def GetDesignPatchLocDict(design_dir, opts):
    design_patch_loc_dict = dict()
    for j, design_name in enumerate(sorted(os.listdir(design_dir))):
        design_name = design_name.split('.')[0]
        design_img = cv2.imread(os.path.join(design_dir, design_name + '.png'), 0)
        design_img = cv2.resize(design_img, None, fx=opts['resize_scale'], fy=opts['resize_scale'])
        design_patch_locs, design_patch_imgs = GetDesignPatches(design_img, size=opts['patch_size'], stride=opts['design_patch_stride'])
        design_patch_loc_dict[design_name] = design_patch_locs
    return design_patch_loc_dict


def GetSherdPatches(img, size, stride):
    (h, w) = img.shape
    p = int(size / 2)
    patch_locs, patch_imgs = list(), list()
    dst_box = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]).astype(np.float32)
    for x in range(0, w - size + 1, stride):
        for y in range(0, h - size + 1, stride):
            for angle in range(0, 360, 10):
                src_box = dst_box + np.array([x, y])
                rotate_mat = cv2.getRotationMatrix2D((x + p, y + p), angle, 1.0)
                src_box = cv2.transform(np.expand_dims(src_box, axis=1), rotate_mat)
                src_box = src_box.squeeze().astype(np.float32)
                aff_mat = cv2.getAffineTransform(src_box[0:3], dst_box[0:3])
                tmp_patch = cv2.warpAffine(img, aff_mat, (size, size))
                if np.mean(tmp_patch) > 10:
                    patch_locs.append([x, y, angle, np.mean(tmp_patch)])
                    patch_imgs.append(tmp_patch)
                # else:
                #     marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                #     cv2.drawContours(marked_img, [src_box[:,np.newaxis,:].astype(np.int)], 0, (0, 255, 0), 3)
                #     cv2.imshow('marked_img', marked_img)
                #     cv2.imshow('tmp_patch', tmp_patch)
                #     cv2.imshow('img', img)
                #     cv2.waitKey()

    return np.array(patch_locs), np.array(patch_imgs)


def GetDesignPatches(img, size, stride):
    (h, w) = img.shape
    patch_locs = list()
    patch_imgs = list()
    for x in range(0, w - size + 1, stride):
        for y in range(0, h - size + 1, stride):
            tmp_patch = img[y:y + size, x:x + size]
            patch_locs.append([x, y])
            patch_imgs.append(tmp_patch)

            # marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # box = np.array([[x, y], [x+size, y], [x+size, y+size], [x, y+size]]).astype(np.float32)
            # cv2.drawContours(marked_img, [box[:,np.newaxis,:].astype(np.int)], 0, (0, 255, 0), 3)
            # cv2.circle(vis_img, (x+int(size/2),y+int(size/2)), 2, (0,0,255), -1)
            # vis_img[y+int(size/2), x+int(size/2)] = [0,255,0]
            # cv2.imshow('tmp_patch', tmp_patch)
            # cv2.imshow('img', marked_img)
            # cv2.waitKey()

    return np.array(patch_locs), np.array(patch_imgs)


def GetBoxOnImg(img, x, y, angle, size):
    (h, w) = img.shape
    p = int(size / 2)
    top_left_x, top_left_x = 0, 0
    if h < size or w < size:
        pad_img = np.zeros((max(h, size), max(w, size)), dtype=np.uint8)
        (hp, wp) = pad_img.shape
        top_left_x = int(wp / 2) - int(w / 2)
        top_left_y = int(hp / 2) - int(h / 2)
        pad_img[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = img
        img = pad_img.copy()

    dst_box = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]).astype(np.float32)
    src_box = dst_box + np.array([x, y])
    rotate_mat = cv2.getRotationMatrix2D((x + p, y + p), angle, 1.0)
    src_box = cv2.transform(np.expand_dims(src_box, axis=1), rotate_mat)
    box_on_ori_img = src_box - np.array([top_left_x, top_left_x])

    src_box = src_box.squeeze().astype(np.float32)
    aff_mat = cv2.getAffineTransform(src_box[0:3], dst_box[0:3])
    patch_img = cv2.warpAffine(img, aff_mat, (size, size))

    marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(marked_img, [src_box[:, np.newaxis, :].astype(np.int)], 0, (255, 0, 255), 3)

    return patch_img, marked_img, box_on_ori_img.squeeze()


class Matcher():
    def __init__(self, opts, sim_model, design_dir):
        cpu_index = faiss.IndexFlatL2(opts['feat_len'])
        self.gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        self.sim_model = sim_model
        self.opts = opts
        self.design_patch_loc_dict = GetDesignPatchLocDict(design_dir, opts)

    def img2feat(self, img):
        img = np.transpose(img, (1,2,0))
        img = cv2.resize(img, (self.opts['patch_size'], self.opts['patch_size']))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2,0,1))
        img = img.astype(float)
        img = img - 128
        img = img / 255
        img = np.expand_dims(img, axis=1)
        self.sim_model.blobs['data'].reshape(*img.shape)
        self.sim_model.blobs['data'].data[...] = img
        self.sim_model.forward()
        feat = self.sim_model.blobs['conv4'].data.copy()
        return feat.reshape(img.shape[0], -1)

    def GetSherdToDesignPatchDists(self, sherd_patch_imgs, design_patch_feats):
        sherd_patch_feats = np.zeros((0, self.opts['feat_len']), dtype=np.float32)
        k = 0
        while k < sherd_patch_imgs.shape[0]:
            sherd_patch_feats = np.vstack((sherd_patch_feats, self.img2feat(sherd_patch_imgs[k:k + self.opts['batch_size']])))
            k += self.opts['batch_size']
        self.gpu_index.reset()
        self.gpu_index.add(design_patch_feats.astype(np.float32))
        sherd2design_dist, sherd2design_idx = self.gpu_index.search(sherd_patch_feats, 1)
        return sherd2design_dist.squeeze(), sherd2design_idx.squeeze()


    def GetTopKMatch(self, sherd_name, depth_img, curve_img, mask_img, design_dir):
        resize_scale = self.opts['resize_scale']
        batch_size = self.opts['batch_size']
        patch_size = self.opts['patch_size']
        sherd_patch_stride = self.opts['sherd_patch_stride']
        design_patch_stride = self.opts['design_patch_stride']
        feat_len = self.opts['feat_len']
        top_k = self.opts['top_k']

        sherd_img = curve_img
        sherd_img = cv2.resize(sherd_img, None, fx=resize_scale, fy=resize_scale)
        sherd_patch_stride = max(sherd_patch_stride, int(max(sherd_img.shape) / 30))
        patch_size = min(patch_size, sherd_img.shape[0], sherd_img.shape[1])
        round_mask = np.zeros((patch_size, patch_size), np.uint8)
        round_mask = cv2.circle(round_mask, (patch_size // 2, patch_size // 2), patch_size // 2, 255, -1)
        sherd_patch_locs, sherd_patch_imgs = GetSherdPatches(sherd_img, size=patch_size, stride=sherd_patch_stride)
        sherd_patch_imgs[:, round_mask == 0] = 0

        # One pair of patches for each design
        candi_matches = list()
        for j, design_name in enumerate(os.listdir(design_dir)):
            design_name = design_name.split('.')[0]
            design_img = cv2.imread(os.path.join(design_dir, design_name + '.png'), 0)
            design_img = cv2.resize(design_img, None, fx=resize_scale, fy=resize_scale)
            design_patch_locs, design_patch_imgs = GetDesignPatches(design_img, size=patch_size, stride=design_patch_stride)
            design_patch_imgs[:, round_mask == 0] = 0
            design_patch_feats = np.zeros((0, feat_len), dtype=np.float32)
            k = 0
            while k < design_patch_imgs.shape[0]:
                design_patch_feats = np.vstack((design_patch_feats, self.img2feat(design_patch_imgs[k:k + batch_size])))
                k += batch_size
            sherd2design_dist, sherd2design_idx = self.GetSherdToDesignPatchDists(sherd_patch_imgs, design_patch_feats)
            sherd2design_dist = np.sqrt(sherd2design_dist) / sherd_patch_locs[:, 3]
            match_score = 1.0 / np.min(sherd2design_dist)
            sherd_patch_idx = np.argmin(sherd2design_dist)
            design_patch_idx = int(sherd2design_idx[sherd_patch_idx])
            candi_matches.append([design_name, sherd_patch_idx, design_patch_idx, match_score])
        candi_matches = sorted(candi_matches, key=lambda match: match[-1], reverse=True)

        top_k_match = list()
        for k in range(0, top_k):
            [design_name, sherd_patch_idx, design_patch_idx, match_score] = candi_matches[k]
            [dx, dy, angle, _] = sherd_patch_locs[sherd_patch_idx]
            sherd_patch, marked_sherd, box_on_sherd = GetBoxOnImg(sherd_img, dx, dy, angle, size=patch_size)
            [dx, dy] = self.design_patch_loc_dict[design_name][design_patch_idx]
            design_img = cv2.imread(os.path.join(design_dir, design_name + '.png'), 0)
            design_img = cv2.resize(design_img, None, fx=resize_scale, fy=resize_scale)
            design_patch, marked_design, box_on_design = GetBoxOnImg(design_img, dx, dy, 0, size=patch_size)
            top_k_match.append({'sherd_id': sherd_name,
                                'match_rank': k + 1,
                                'match_score': match_score,
                                'design_id': design_name,
                                'rect_on_sherd': box_on_sherd,
                                'rect_on_design': box_on_design,
                                'sherd_depth': depth_img,
                                'sherd_curve': 255 - sherd_img,
                                'design_img': 255 - design_img,
                                'cropped_sherd_patch': 255 - sherd_patch,
                                'cropped_design_patch': 255 - design_patch,
                                'marked_sherd': 255 - marked_sherd,
                                'marked_design': 255 - marked_design})
        return top_k_match


    def WriteMatchResults(self, top_k_match, save_dir):
        sherd_id, sherd_depth, sherd_curve = \
            top_k_match[0]['sherd_id'], top_k_match[0]['sherd_depth'], top_k_match[0]['sherd_curve']
        tmp_save_dir = os.path.join(save_dir, sherd_id)
        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)
        cv2.imwrite(os.path.join(tmp_save_dir, 'sherd_depth.png'), sherd_depth)
        cv2.imwrite(os.path.join(tmp_save_dir, 'sherd_curve.png'), sherd_curve)

        for i, match in enumerate(top_k_match):
            match_rank, match_score, design_id, design_img, rect_on_sherd, rect_on_design, \
            cropped_sherd_patch, cropped_design_patch, marked_sherd, marked_design = \
                match['match_rank'], match['match_score'], match['design_id'], match['design_img'], match['rect_on_sherd'], match['rect_on_design'], \
                match['cropped_sherd_patch'], match['cropped_design_patch'], match['marked_sherd'], match['marked_design']
            rank_dir = os.path.join(tmp_save_dir, str(match_rank))
            if not os.path.exists(rank_dir):
                os.mkdir(rank_dir)
            cv2.imwrite(os.path.join(rank_dir, 'design_img.png'), design_img)
            cv2.imwrite(os.path.join(rank_dir, 'cropped_sherd_patch.png'), cropped_sherd_patch)
            cv2.imwrite(os.path.join(rank_dir, 'cropped_design_patch.png'), cropped_design_patch)
            cv2.imwrite(os.path.join(rank_dir, 'marked_sherd.png'), marked_sherd)
            cv2.imwrite(os.path.join(rank_dir, 'marked_design.png'), marked_design)

        frontend_xls = xlsxwriter.Workbook(os.path.join(tmp_save_dir, 'frontend.xlsx'))
        frontend_table = frontend_xls.add_worksheet()
        center = frontend_xls.add_format({'align': 'center'})
        frontend_table.set_column('A:A', 15)
        frontend_table.set_column('B:B', 15)
        frontend_table.set_column('C:C', 15)
        frontend_table.set_column('D:D', 15)
        frontend_table.set_column('E:E', 15)
        frontend_table.write(0, 0, 'match_rank', center)
        frontend_table.write(0, 1, 'match_score', center)
        frontend_table.write(0, 2, 'design_id', center)
        frontend_table.write(0, 3, 'rect_on_sherd', center)
        frontend_table.write(0, 4, 'rect_on_design', center)
        for i, match in enumerate(top_k_match):
            sherd_id, match_rank, match_score, design_id, rect_on_sherd, rect_on_design = \
                match['sherd_id'], match['match_rank'], match['match_score'], match['design_id'], match['rect_on_sherd'], match['rect_on_design']
            frontend_table.write(i+1, 0, str(match_rank), center)
            frontend_table.write(i+1, 1, "{:.4f}".format(match_score), center)
            frontend_table.write(i+1, 2, design_id, center)
            frontend_table.write(i+1, 3, np.array2string(rect_on_sherd))
            frontend_table.write(i+1, 4, np.array2string(rect_on_design))
        frontend_xls.close()


        final_xls = xlsxwriter.Workbook(os.path.join(tmp_save_dir, 'final.xlsx'))
        final_table = final_xls.add_worksheet()
        center = final_xls.add_format({'align': 'center', 'valign': 'vcenter',})
        final_table.set_column('A:A', 15)
        final_table.set_column('B:B', 15)
        final_table.set_column('C:C', 15)
        final_table.set_column('D:D', 35)
        final_table.set_column('E:E', 25)
        final_table.set_column('F:F', 25)
        final_table.set_column('G:G', 35)
        final_table.set_column('H:H', 40)
        final_table.write(0, 0, 'match_rank', center)
        final_table.write(0, 1, 'match_score', center)
        final_table.write(0, 2, 'design_id', center)
        final_table.write(0, 3, 'sherd_depth')
        final_table.write(0, 4, 'sherd_patch')
        final_table.write(0, 5, 'design_patch')
        final_table.write(0, 6, 'marked_sherd')
        final_table.write(0, 7, 'marked_design')
        for i, match in enumerate(top_k_match):
            final_table.set_row(i+1, 200)
            match_rank, match_score, design_id, design_img, rect_on_sherd, rect_on_design, \
            cropped_sherd_patch, cropped_design_patch, marked_sherd, marked_design = \
                match['match_rank'], match['match_score'], match['design_id'], match['design_img'], match['rect_on_sherd'], match['rect_on_design'], \
                match['cropped_sherd_patch'], match['cropped_design_patch'], match['marked_sherd'], match['marked_design']
            final_table.write(i+1, 0, str(match_rank), center)
            final_table.write(i+1, 1, "{:.4f}".format(match_score), center)
            final_table.write(i+1, 2, design_id, center)

            scale = 200.0 / max(sherd_depth.shape)
            y_offset = (250.0 - scale * sherd_depth.shape[0]) * 0.8 / 2
            final_table.insert_image(i+1, 3, os.path.join(tmp_save_dir, 'sherd_depth.png'), {'x_scale': scale, 'y_scale': scale, 'x_offset': 0, 'y_offset': y_offset})

            scale = 150.0 / max(cropped_sherd_patch.shape)
            y_offset = (250.0 - scale * cropped_sherd_patch.shape[0]) * 0.8 / 2
            final_table.insert_image(i+1, 4, os.path.join(tmp_save_dir, str(match_rank), 'cropped_sherd_patch.png'), {'x_scale': scale, 'y_scale': scale, 'x_offset': 0, 'y_offset': y_offset})
            final_table.insert_image(i+1, 5, os.path.join(tmp_save_dir, str(match_rank), 'cropped_design_patch.png'), {'x_scale': scale, 'y_scale': scale, 'x_offset': 0, 'y_offset': y_offset})

            scale = 200.0 / max(marked_sherd.shape)
            y_offset = (250.0 - scale * marked_sherd.shape[0]) * 0.8 / 2
            final_table.insert_image(i+1, 6, os.path.join(tmp_save_dir, str(match_rank), 'marked_sherd.png'), {'x_scale': scale, 'y_scale': scale, 'x_offset': 0, 'y_offset': y_offset})

            scale = 250.0 / max(marked_design.shape)
            y_offset = (250.0 - scale * marked_design.shape[0]) * 0.8 / 2
            final_table.insert_image(i+1, 7, os.path.join(tmp_save_dir, str(match_rank), 'marked_design.png'), {'x_scale': scale, 'y_scale': scale, 'x_offset': 0, 'y_offset': y_offset})

        final_xls.close()
