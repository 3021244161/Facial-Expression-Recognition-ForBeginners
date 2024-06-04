import matplotlib.pyplot as plt

# 读取日志文件
epochs = []
loss = []
accuracy = []
val_loss = []
val_accuracy = []

with open('training_log_vgg19_100epo.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        print(f"Parsing line: {line.strip()}")
        print(f"Split parts: {parts}")
        try:
            if len(parts) == 4:
                epoch_num = int(parts[0].split(' ')[1][:-1])
                train_loss = float(parts[0].split('=')[1])
                train_acc = float(parts[1].split('=')[1])
                valid_loss = float(parts[2].split('=')[1])
                valid_acc = float(parts[3].split('=')[1])
                
                epochs.append(epoch_num)
                loss.append(train_loss)
                accuracy.append(train_acc)
                val_loss.append(valid_loss)
                val_accuracy.append(valid_acc)
            else:
                print(f"Line format is incorrect: {line.strip()}")
        except (IndexError, ValueError) as e:
            print(f"Error parsing line: {line.strip()}. Error: {e}")

# 绘制loss曲线
plt.figure(figsize=(12, 6))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制accuracy曲线
plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
