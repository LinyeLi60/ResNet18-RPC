
import pysftp
import matplotlib.pyplot as plt


cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

with pysftp.Connection('101.132.109.187', port=6000, username='lilinye', password='168518', cnopts=cnopts) as sftp:
    # sftp.cd('/home/lilinye/PythonProjects/deeplearning/resnet18_checkout/tools/weights')
    files = sftp.listdir('/home/lilinye/PythonProjects/deeplearning/resnet18_checkout/tools/weights')
    losses = []
    for file in files:
        losses.append(float(file.split('_')[-1].strip('.pth')))
    print(losses)

    fig = plt.figure()
    plt.plot(losses, color='r')  # 绘制黑色虚线
    plt.savefig('loss.png')

