
import matplotlib.pyplot as plt
import os

save_dir = r'G:\Research\EEG_Project\Template\CodeDir\tools\训练和验证的损失曲线'
# 训练和验证的精确度和损失曲线
def plot_metrics(train_losses,val_losses,train_accuracies,val_accuracies):
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))

    # # 保存为 CSV：记录损失与精度值
    # os.makedirs(csv_dir, exist_ok=True)
    # csv_path = os.path.join(save_dir, 'training_validation_metrics.csv')
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # header = [f'Run_{timestamp}_Epochs']
    # # 构造 DataFrame，并保留两位小数
    # metrics_df = pd.DataFrame({
    #     'Train_Loss': np.round(self.train_losses, 2),
    #     'Train_Accuracy': np.round(self.train_accuracies, 2),
    #     'Val_Loss': np.round(self.val_losses, 2),
    #     'Val_Accuracy': np.round(self.val_accuracies, 2)
    # })
    # # 添加空行和标题
    # with open(csv_path, 'a', encoding='utf-8-sig') as f:
    #     f.write('\n')
    #     f.write(','.join(header) + '\n')
    # # 写入 CSV 文件（保留两位小数）
    # metrics_df.to_csv(csv_path, mode='a', index_label='Epoch', float_format='%.2f', encoding='utf-8-sig')

    plt.close()