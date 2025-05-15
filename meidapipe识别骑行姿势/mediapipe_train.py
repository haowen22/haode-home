import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, utils, callbacks
import joblib
import os


# 数据加载和预处理（适配视频提取的特征格式）
def load_data():
    try:
        # 加载预处理好的特征
        X = np.load("X_features.npy")
        y = np.load("y_labels.npy")

        # 如果文件不存在，尝试加载训练集/测试集
        if not os.path.exists("X_train.npy"):
            # 标签编码（如果尚未编码）
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # 保存划分好的数据集
            np.save("X_train.npy", X_train)
            np.save("X_test.npy", X_test)
            np.save("y_train.npy", y_train)
            np.save("y_test.npy", y_test)
            joblib.dump(encoder, "label_encoder.pkl")
        else:
            # 直接加载已划分的数据
            X_train = np.load("X_train.npy")
            X_test = np.load("X_test.npy")
            y_train = np.load("y_train.npy")
            y_test = np.load("y_test.npy")
            encoder = joblib.load("label_encoder.pkl")

        # 归一化处理
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 将标签转换为one-hot编码
        y_train = utils.to_categorical(y_train)
        y_test = utils.to_categorical(y_test)

        return X_train, X_test, y_train, y_test, scaler, encoder

    except FileNotFoundError as e:
        print(f"错误：找不到数据文件！请先运行特征提取代码。\n{e}")
        return None, None, None, None, None, None


# 增强型模型架构（适配视频特征维度）
def build_video_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape,
                     kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    return model


# 可视化函数
def plot_training_history(history):
    plt.figure(figsize=(15, 5))

    # 准确率曲线
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # AUC曲线
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


if __name__ == "__main__":
    # 加载数据
    X_train, X_test, y_train, y_test, scaler, encoder = load_data()

    if X_train is not None:
        # 获取类别名称
        class_names = list(encoder.classes_)
        print("检测到的类别:", class_names)

        # 构建模型
        model = build_video_model(input_shape=(X_train.shape[1],),
                                  num_classes=len(class_names))

        # 早停机制
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            mode='max',
            restore_best_weights=True
        )

        # 学习率调度
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stop, lr_scheduler],
            verbose=1
        )

        # 可视化训练过程
        plot_training_history(history)

        # 模型评估
        print("\n测试集评估结果：")
        results = model.evaluate(X_test, y_test, verbose=0)
        for name, value in zip(model.metrics_names, results):
            print(f"{name}: {value:.4f}")

        # 生成分类报告
        y_pred = model.predict(X_test).argmax(axis=1)
        y_true = y_test.argmax(axis=1)
        print("\n分类报告：")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # 绘制混淆矩阵
        plot_confusion_matrix(y_true, y_pred, class_names)

        # 保存模型和预处理参数
        model.save("cycling_pose_model.h5")

        # 转换为TFLite格式
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open("cycling_pose_model.tflite", "wb") as f:
            f.write(tflite_model)

        # 保存预处理参数
        np.save("scaler_params.npy", {
            'min': scaler.min_,
            'scale': scaler.scale_
        })

        print("\n模型已保存为 cycling_pose_model.h5 和 cycling_pose_model.tflite")