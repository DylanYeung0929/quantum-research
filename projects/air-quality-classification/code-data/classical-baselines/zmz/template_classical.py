import os
import math
import numpy as np
import pandas as pd
import pyvqnet
from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.tensor import tensor
from sklearn.metrics import f1_score
from pyvqnet.data.data import data_generator
from pyvqnet.nn.module import Module
from pyvqnet.nn import Linear, ReLu
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CrossEntropyLoss
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_data():
    """
    1. 读取数据并转换为张量
    """
    # 读取四个文档中的数据并转换为张量
    X_train_raw = pd.read_csv(train_features_file)
    X_test_raw = pd.read_csv(test_features_file)
    train_df_original = pd.read_csv(original_train_file)
    test_df_original = pd.read_csv(original_test_file)
    # 从原始测试集中获取标签
    y_train_raw = train_df_original['Air Quality']
    if 'Air Quality' in test_df_original.columns:
        y_test_raw = test_df_original['Air Quality']
        print("加载测试标签成功。")
        if 'Air Quality' in X_test_raw.columns:
            X_test_raw = X_test_raw.drop('Air Quality', axis=1)
            print("已从加载的X_test_raw中移除'Air Quality'列。")
    else:
        print("Warning:原始test_data.csv中未找到'Air Quality'列。无法在测试集上评估。")
        exit("无法在无标签的测试集上评估，请检查 test_data.csv 或调整代码。")

    # 进行标准化缩放处理，使其均值为0，标准差为1
    # 必须用训练集的scalar来处理测试集，保证相同的缩放标准
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 将标签字符串转化为数值型
    # LabelEncoder会按照字母顺序对标签进行编码('Good'->0,'Hazardous'->1,'Moderate'->2,'Poor'->3)影响不大
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw) 

    # 打印标签类别映射以确认进度
    print("\n标签类别映射:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"{class_name} -> {i}")

    # 将数据转换为Numpy数组,flatten()将y_train和y_test转换为一维数组
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64).flatten()
    y_test = np.array(y_test, dtype=np.int64).flatten()

    return X_train, y_train, X_test, y_test, encoder




class AirQualityNN(Module):
    """
    2. 在这里完成初始化神经网络模型的代码
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AirQualityNN, self).__init__()
        # Define network layers
        self.fc1 = Linear(input_dim, hidden_dim) # Input layer to hidden layer
        # Change activation function to ReLu
        self.relu = ReLu()
        self.fc2 = Linear(hidden_dim, output_dim) # Hidden layer to output layer

    def forward(self, x):
        # Define forward pass path
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def model_train(X_train, y_train, encoder):
    """
    3. 在这里完成训练模型的代码
    """
 
    # 定义模型参数（可调整）
    input_dim = X_train.shape[1] # 输入特征数量
    output_dim = len(encoder.classes_) # 输出类别数量
    # 实例化模型 model
    model = AirQualityNN(input_dim, hidden_dim, output_dim)

    # 获取模型参数量并输出呈现
    total_params_model = sum(p.size for p in model.parameters())
    print(f"模型总参数量: {total_params_model}")
    if total_params_model <= 200:
        print("参数量符合要求 (<= 200)")
    else:
        print("警告：参数量超出限制 (> 200)")
    
    # 使用Adam优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)
    training_successful = True

    print("开始训练 —— 使用手动批处理和手动损失计算...")
    num_train_samples = X_train.shape[0] # 获取训练集样本数量
    num_batches = math.ceil(num_train_samples / batch_size) # 计算批次数（总样本除以批次大小）

    for epoch in range(epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0.0 # 初始化损失
        correct_train = 0 # 初始化正确预测数量
        total_train = 0 # 初始化总样本数量

        # 每轮训练开始前打乱训练数据顺序，避免模型记忆数据顺序
        permutation = np.random.permutation(num_train_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for batch_idx in range(num_batches):
            # 对于每个batch，计算当前批次的起始和结束索引
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train_samples)
            data_np = X_train_shuffled[start_idx:end_idx] # float32类型
            target_np = y_train_shuffled[start_idx:end_idx] # int64类型
            current_batch_size = target_np.shape[0] # 当前批次大小

            # 将数据转换为QTensor类型
            data = QTensor(data_np) # float32类型
            target = QTensor(target_np) # int64类型

            # 前向传播
            optimizer.zero_grad()   # 清除上一轮的梯度
            outputs = model(data)   # 前向传播，得到 logits
            # 确保输出的dtype为float32/64
            outputs_np = outputs.numpy()
            if outputs_np.dtype != np.float32 and outputs_np.dtype != np.float64:
                # print(f"  警告: Outputs dtype is {outputs_np.dtype} in Epoch {epoch+1}, Batch {batch_idx}. Casting to float32.")
                outputs_np = outputs_np.astype(np.float32)
                outputs = QTensor(outputs_np) # 将输出转换为QTensor类型
            
            # ----- 交叉熵损失（方法一）-----
            # 对模型的输出outputs形状为[batch_size, num_classes]应用log-softmax，得到每个类别的对数概率
            # log_probs = pyvqnet.tensor.log_softmax(outputs)
            # 将目标标签从[batch_size]转换为[batch_size, 1]以便gather操作
            # target_reshaped = target.reshape((current_batch_size, 1))
            # # 使用gather操作从log_probs中提取每个样本的对数概率
            # gathered_log_probs = pyvqnet.tensor.gather(log_probs, dim=1, index=target_reshaped)
            # # 对所有样本的 -log(p) 值取平均
            # loss = -pyvqnet.tensor.mean(gathered_log_probs)

            # ----- 交叉熵损失（方法一）-----
            target_reshaped = target.reshape((current_batch_size, 1))
            loss_result = CrossEntropyLoss()
            loss = loss_result(target_reshaped,outputs)

            # 健壮性检查，确保损失张量能参与计算图和反向传播
            if not loss.requires_grad:
                print(f"Warning: Batch {batch_idx} - Loss does not require grad! Graph broken.")
                training_successful = False
                break

            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新参数
            running_loss += loss.item() # 累加损失、

            # 计算训练精度 Accuracy
            # 表示暂时不记录计算图，结婚谁能内存，提高速度，只用于评估
            with pyvqnet.no_grad():
                # 再次前向传播，得到logits
                pred_outputs = model(data)
                # 利用argmax函数获取预测的类别索引
                predicted_train = pyvqnet.tensor.argmax(pred_outputs, dim=1)
            target_1d_np = target.numpy()
            total_train += current_batch_size
            # 统计预测正确的样本数量
            correct_train += (predicted_train.numpy() == target_1d_np).sum()
            
            # 计算平均损失和轮次准确率
            epoch_loss = running_loss / num_batches if num_batches > 0 else 0
            epoch_acc_train = 100. * correct_train / total_train if total_train > 0 else 0
            # 每10个epoch打印一次训练损失和准确率
        if (epoch + 1) % 25 == 0 or epoch == epochs-1:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc_train:.2f}%')

    if training_successful:
        print("训练完成！")
    else:
        print("训练因错误提前终止。")
    return model

def model_test(model):
    """
    4. 使用测试数据集验证模型
    """
    print("开始在测试集上评估...")
    model.eval() # 设置模型到评估模式

    # 用于存放预测值与真实标签，用于后续计算比较
    all_preds = []
    all_targets = []
    all_targets = y_test

    X_test_tensor = QTensor(X_test) # float32类型
    with pyvqnet.no_grad():
        outputs_test = model(X_test_tensor) # 前向传播，得到logits
        try:
            predicted = pyvqnet.tensor.argmax(outputs_test, dim=1) # 获取预测的类别索引
        except Exception as e:
            print(f"Error during argmax operation: {e}")
            exit("无法计算argmax,请检查模型输出。")
        except ValueError as ve:
            print(f"评估期间计算预测时发生 ValueError: {ve}")
            predicted = QTensor(np.array([]))
        except AttributeError as ae:
            print(f"评估期间计算预测时发生 AttributeError (可能缺少 argmax): {ae}")
            predicted = QTensor(np.array([]))

        # 将预测结果转换为numpy数组
        if predicted is not None and predicted.size > 0:
            all_preds = predicted.numpy()
        else:
            print("警告：预测结果为空或无效。")
            all_preds = np.array([])

    # 仅在预测和真实标签都存在且长度相等时计算F1分数
    if len(all_preds) > 0 and len(all_targets) > 0 and len(all_preds) == len(all_targets):
        Accuracy = np.sum(all_preds == all_targets) / len(all_targets) * 100.0
        f1 = f1_score(all_targets, all_preds, average='macro',zero_division=1)
        print(f"测试集准确率Accuracy: {Accuracy:.4f}%")
        print(f"测试集F1分数: {f1:.4f}")

        # with open(output_results_file, 'w') as f:
        #     f.write(f"Accuracy={Accuracy:.4f}\n")
        #     f.write(f"Average F1 Score={f1:.4f}\n")
        # print(f"评估结果已保存到: {output_results_file}")
        return Accuracy, f1
    else:
        print("评估跳过，因为预测结果为空或与目标标签不匹配。")
    


if __name__ == "__main__":

    data_directory = r"D:\360MoveData\Users\wzj31\Desktop\Quantum Computing\复赛最终版代码"
    output_results_file = os.path.join(data_directory, "classical_results.txt")
    # 两个处理过后的文件路径
    train_features_file = os.path.join(data_directory, "preprocessed_train_features_zero_negative_rows_deleted.csv")
    test_features_file = os.path.join(data_directory, "preprocessed_test_features_zero_negative_rows_deleted.csv")
    # 两个原始文件路径
    original_train_file = os.path.join(data_directory, "train_data.csv")
    original_test_file = os.path.join(data_directory, "test_data.csv")

    # 超参数模型
    hidden_dim = 10 # 隐藏层神经元数量
    learning_rate = 0.01 # 学习率
    batch_size = 64 # 批量大小
    epochs = 75 # 训练轮数

    X_train, y_train, X_test, y_test, encoder=load_data()

    accuracy_list=[0]*10 # 用于存储每次训练的准确率
    f1_list = [0]*10

    for i in range(10):
        print(f"第{i+1}次训练")
        # 训练模型
        model=model_train(X_train, y_train, encoder)
        accuracy_list[i-1],f1_list[i-1] = model_test(model)
        mean_accuracy = np.mean(accuracy_list)
        mean_f1 = np.mean(f1_list)
    print(f"平均准确率: {mean_accuracy:.4f}%")
    print(f"平均F1分数: {mean_f1:.4f}")
        
