import os
import tensorflow as tf
from BertLayer import Bert
from Data.data import DataGenerator
from Loss.loss import BERT_Loss
from Loss.utils import calculate_pretrain_task_accuracy
from config import Config
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

nsp_accuracy = tf.keras.metrics.Accuracy()
mlm_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Bert(Config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    loss_fn = BERT_Loss()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint(Config['Saved_Weight']))
    manager = tf.train.CheckpointManager(checkpoint, directory=Config['Saved_Weight'], max_to_keep=5)
    dataset = DataGenerator(Config)
    log_dir = os.path.join(Config['Log_Dir'], datetime.now().strftime("%Y-%m-%d"))
    writer = tf.summary.create_file_writer(log_dir)
    EPOCH = 50

    # 定义训练步骤函数
    @tf.function
    def train_step(batch_x, batch_mlm_mask, origin_x, batch_segment, batch_padding_mask, batch_y):
        with tf.GradientTape() as t:
            nsp_predict, mlm_predict, sequence_output = model((batch_x, batch_padding_mask, batch_segment),
            training=True)
            nsp_loss, mlm_loss = loss_fn((mlm_predict, batch_mlm_mask, origin_x, nsp_predict, batch_y))
            nsp_loss = tf.reduce_mean(nsp_loss)
            mlm_loss = tf.reduce_mean(mlm_loss)
            loss = nsp_loss + mlm_loss
        gradients = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                # 更新评估指标
        return loss, nsp_predict, mlm_predict,nsp_loss,mlm_loss

    # 在策略的范围内进行训练
    with writer.as_default():
        for epoch in range(EPOCH):
            for step, (batch_x, batch_mlm_mask, origin_x, batch_segment, batch_padding_mask, batch_y) in enumerate(dataset):
                loss, nsp_predict, mlm_predict,nsp_loss,mlm_loss = strategy.run(train_step, args=(batch_x, batch_mlm_mask, origin_x, batch_segment, batch_padding_mask, batch_y))
        
                # 执行其他计算或记录日志的操作
                print("epoch:{},nsp_loss:{},mlm_loss:{}".format(epoch+1,nsp_loss,mlm_loss))

            # 保存模型
            if epoch % 10 == 0:
                manager.save()
