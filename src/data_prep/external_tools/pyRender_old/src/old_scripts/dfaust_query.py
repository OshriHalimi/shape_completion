import re
from pprint import pprint
from pathlib import Path
from dfaust_utils import banner,file_size

from copy import deepcopy


# sids = ['50002', '50004', '50007', '50009', '50020',
#         '50021', '50022', '50025', '50026', '50027']

# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

def generate_dfaust_map():
    fp = Path() / "subjects_and_sequences.txt"
    assert fp.is_file(), "Could not find Dynamic Faust subjects_and_sequences.txt file"
    print(f'Found dfaust subjects and sequence list at {fp.parents[0].absolute()}')
    lines = [line.rstrip('\n') for line in open(fp)]
    # pprint(lines)

    sub_list = []
    last_subj = None

    for line in lines:
        m = re.match(r"(\d+)\s+\((\w+)\)", line)
        if m:  # New hit
            id, gender = m.group(1, 2)
            last_subj = DFaustSubject(id, gender)
            sub_list.append(last_subj)
        elif line.strip():
            seq, frame_cnt = line.split()
            last_subj.seq_grp.append(seq)
            last_subj.frame_cnts.append(int(frame_cnt))

    return DFaustMap(sub_list)


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

class DFaustMap:
    def __init__(self, subject_list):
        # TODO - Encoding is currently relative to the dataset. Maybe better to do it absolute
        # Compute all unique sequence names
        used = set()
        seqs = [seq for sub in subject_list for seq in sub.seq_grp]
        uniq_seqs = [x for x in seqs if x not in used and (used.add(x) or True)]

        # Encoding
        seq2eseq = dict(zip(uniq_seqs, range(len(uniq_seqs))))

        # Encode
        for i, sub in enumerate(subject_list):
            sub.encoded_id = i
            sub.encoded_seq_grp = [seq2eseq[seq] for seq in sub.seq_grp]

        id_list = [sub.id for sub in subject_list]  # Presumed unique
        id2eid = dict(zip(id_list, range(len(id_list))))

        # Mappers
        self.seq2eseq = seq2eseq
        self.eseq2seq = {v: k for k, v in seq2eseq.items()}
        self.id2eid = id2eid
        self.eid2id = {v: k for k, v in id2eid.items()}
        self.subject_list = subject_list

        # Hacky variables
        self.num_angs = None
        self.world2cam_mats = None

    # Getters
    def disk_size(self):
        return file_size(self.num_frames()*648*1024) #Around 648 KB per mesh

    def num_subjects(self):
        return len(self.subject_list)

    def num_sequences(self):
        return len(self.seq_names())

    def num_frames(self):
        n_frames = 0
        for sub in self.subject_list:
            n_frames += sum(sub.frame_cnts)
        return n_frames

    def subjects(self):
        return self.subject_list

    def seq_names(self):
        return list(self.seq2eseq.keys())

    def sub_names(self):
        return list(self.id2eid.keys())

    def sub_by_id(self, ids):
        if type(ids) is not list: ids = [ids]

        ret = []
        for sub in self.subject_list:
            if sub.id in ids:
                ret.append(sub)
        return ret

    def sub_by_eid(self, eids):
        if type(eids) is not list: ids = [eids]
        return [self.subject_list[eid] for eid in eids]

    # Encoding translation
    def seq_by_eseq(self, eseqs):
        if type(eseqs) is not list: eseqs = [eseqs]
        return [self.eseq2seq[eseq] for eseq in eseqs]

    def eseq_by_seq(self, seqs):
        if type(seqs) is not list: seqs = [seqs]
        return [self.seq2eseq[seq] for seq in seqs]

    def id_by_eid(self, eids):
        if type(eids) is not list: eids = [eids]
        return [self.eid2id[eid] for eid in eids]

    def eid_by_id(self, ids):
        if type(ids) is not list: ids = [ids]
        return [self.id2eid[id] for id in ids]

    # Filter
    def filter_by_seq(self, required_seqs, narrow=False):
        # Handles either encoded or decoded. If narrow = True -> Leaves only the requires sequences alive
        if type(required_seqs) is not list: required_seqs = [required_seqs]

        # Handle encoding
        required_seqs = [seq if str(seq).isdigit() else self.seq2eseq[seq] for seq in required_seqs]
        required_eseqs = sorted([int(seq) for seq in required_seqs]) # Actually holds eseqs
        # pprint(required_seqs)
        new_subs = []
        if narrow:
            required_seqs = self.seq_by_eseq(required_eseqs)  # Translate to strings

        for sub in self.subject_list:
            if all(elem in sub.encoded_seq_grp for elem in required_eseqs):
                if narrow:
                    new_sub = deepcopy(sub)
                    match_indices = [sub.seq_grp.index(seq) for seq in required_seqs]
                    new_sub.frame_cnts = [new_sub.frame_cnts[i] for i in match_indices]
                    new_sub.seq_grp = required_seqs
                else:
                    new_sub = sub
                new_subs.append(new_sub)

        return DFaustMap(new_subs)

    def filter_by_gender(self, gender):
        assert gender == 'male' or gender == 'female'

        new_subs = []
        for sub in self.subject_list:
            if sub.gender == gender:
                new_subs.append(sub)

        return DFaustMap(new_subs)

    def filter_by_id(self, ids):  # Handles either encoded or decoded
        if type(ids) is not list: ids = [ids]
        # Handle encoding
        ids = [id if int(id)> 1000 else self.eid2id[id] for id in ids] # Presuming knowledge on dataset

        new_subs = []
        for sub in self.subject_list:
            if sub.id in ids:
                new_subs.append(sub)

        return DFaustMap(new_subs)




# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

class DFaustSubject:
    def __init__(self, id, gender):
        self.id = id
        self.gender = gender
        self.seq_grp = []
        self.frame_cnts = []
        self.encoded_id = None
        self.encoded_seq_grp = []

    def num_sequences(self):
        return len(self.seq_grp)


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#

def unit_test():
    map = generate_dfaust_map()
    print(map.seq_names())
    print(map.sub_names())
    print(map.num_frames())
    map2 = map.filter_by_gender('male')  # Or female
    print(map2.seq_names())
    print(map2.sub_names())
    # banner()
    map2 = map.filter_by_id([0,1])
    print(map2.seq_names())
    print(map2.sub_names())
    banner('Filter by ID')
    map2 = map.filter_by_id(['50002','50004']).filter_by_seq('chicken_wings',narrow=1)
    print(map2.seq_names())
    print(map2.sub_names())

    map3 = map.filter_by_seq(map.seq_names())
    print(map3.seq_names())
    print(map3.sub_names())

    map3 = map.filter_by_seq('chicken_wings')  # Or female
    print(map3.seq_names())
    print(map3.sub_names())
    map3 = map.filter_by_seq(['chicken_wings', 'hips', 'jumping_jacks'])
    print(map3.seq_names())
    print(map3.sub_names())
    map3 = map.filter_by_seq(['0', '1'])
    print(map3.seq_names())
    print(map3.sub_names())
    banner()
    map3 = map.filter_by_seq([0],narrow=True)
    print(map3.num_frames())
    print(map3.num_subjects())
    print(map3.num_sequences())
    print(map3.seq_names())
    print(map3.sub_names())

    print(map.sub_by_id('50004')[0].id)
    print(map.sub_by_id(['50004', '50009']))
    print(map.sub_by_eid([1])[0].id)
    print(map.sub_by_eid([0, 1]))

    print(map.seq_by_eseq(11))
    print(map.eseq_by_seq('chicken_wings'))

    print(map.id_by_eid(0))
    print(map.eid_by_id('50004'))


if __name__ == '__main__':
    unit_test()
