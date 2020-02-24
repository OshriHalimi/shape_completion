import yagmail
import zipfile
from datetime import datetime
import os


class TensorboardEmailer:
    def __init__(self, exp_dp):
        self.sender = "giovanni.cvpr@gmail.com"
        self.password = "cvpr2020" # TODO - This password exists on a public github
        self.to = self.sender
        self.exp_dp = exp_dp
        self.zip_fp = self.exp_dp / "email_package.zip"

    def send_report(self, final_results_str):
        yag = yagmail.SMTP(user=self.sender, password=self.password)
        self.prep_attachment()
        yag.send(
            to=self.to,
            subject=f"[{datetime.now()}] Results for Experiment : {self.exp_dp.name} from {os.environ['COMPUTERNAME']}",
            contents=final_results_str,
            attachments=[self.zip_fp],
        )

    def prep_attachment(self):
        zf = zipfile.ZipFile(self.zip_fp, "w")

        exp_dump_dp = self.exp_dp.parents[0]
        exp_dump_arch = exp_dump_dp.name
        zf.write(exp_dump_dp, exp_dump_arch)

        exp_dp = self.exp_dp
        exp_dp_arch = os.path.join(exp_dump_arch, exp_dp.name)
        zf.write(exp_dp, exp_dp_arch)

        metrics_fp = self.exp_dp / 'metrics.csv'
        metrics_fp_arch = os.path.join(exp_dp_arch, 'metrics.csv')
        zf.write(metrics_fp, metrics_fp_arch)

        meta_fp = self.exp_dp / 'meta_tags.csv'
        meta_fp_arch = os.path.join(exp_dp_arch, 'meta_tags.csv')
        zf.write(meta_fp, meta_fp_arch)

        tf_dp = self.exp_dp / 'tf'
        tf_dp_arch = os.path.join(exp_dp_arch, 'tf')
        zf.write(tf_dp, tf_dp_arch)

        for f in os.listdir(self.exp_dp / 'tf'):
            zf.write(os.path.join(tf_dp,f), os.path.join(tf_dp_arch, f))

        zf.close()
