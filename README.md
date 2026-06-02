# =========================================================

    total = 0

    for images, labels in train_data_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    average_loss = running_loss / len(train_data_loader)

    train_loss.append(average_loss)

    print(

        'Epoch =', epoch + 1,

        'Loss =', average_loss,

        'Accuracy =', accuracy

    )

# =========================================================
# SAVE MODEL
# =========================================================

torch.save(

    model.state_dict(),

    'models/drowsiness_cnn_model.pth'

)

print('Model Saved Successfully!')

# =========================================================
# LOSS GRAPH
# =========================================================

plt.plot(train_loss)

plt.title('Training Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()