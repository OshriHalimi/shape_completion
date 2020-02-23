from util.mesh.ops import trunc_to_vertex_mask
import util.mesh.io


class CompletionSaver:

    def __init__(self, exp_dir, testset_names, extended_save, f):
        from cfg import SAVE_MESH_AS
        self.save_func = getattr(util.mesh.io, f'write_{SAVE_MESH_AS}')
        self.extended_save = extended_save
        self.f = f  # Might be None

        self.dump_dirs = []
        for ts_name in testset_names:
            dp = exp_dir / 'completions' / ts_name
            dp.mkdir(parents=True, exist_ok=True)
            self.dump_dirs.append(dp)

    def save_completions_by_batch(self, pred, b, set_id):
        dump_dp = self.dump_dirs[set_id]

        # TODO - Make this generic, and not key dependent. Insert support for P2P
        gtrb = pred['completion_xyz'].cpu().numpy()
        for i, (gt_hi, tp_hi) in enumerate(zip(b['gt_hi'], b['tp_hi'])):
            gt_hi, tp_hi = '_'.join(str(x) for x in gt_hi), '_'.join(str(x) for x in tp_hi)
            gtr_v = gtrb[i, :, :3]
            self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_res', gtr_v, self.f)

            if self.extended_save:
                gt_v = b['gt'][i, :, :3]
                self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_gt', gt_v, self.f)

                tp_v = b['tp'][i, :, :3]
                self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_tp', tp_v, self.f)

                gt_part_v, gt_part_f = trunc_to_vertex_mask(gt_v, self.f, b['gt_mask'][i])
                self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_gtpart', gt_part_v, gt_part_f)
