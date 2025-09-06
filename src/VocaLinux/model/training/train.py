import os

from callbacks import HistoryCallback, ScheduledSamplingCallback

epochs = 10
init_epoch = 0
final_epoch = init_epoch + epochs
history_save_location = "history.json"
plot_save_path = "training_plot.png"
history_file = "/kaggle/input/tensorflow-listen-attend-and-spell/history.json"

if os.path.exists(history_file):
    try:
        with open(history_file, 'r') as f:
            loaded_history_data = json.load(f)
            loaded_epoch = loaded_history_data.get("last_completed_epoch")
            if loaded_epoch is not None:
                init_epoch = loaded_epoch + 1
                final_epoch = init_epoch + epochs
            print(f"Resuming training from epoch {init_epoch}.")
    except Exception as e:
        print(f"Could not load previous history from {history_file}: {e}")
        print(f"Starting from default value: {init_epoch}")

training_mode = True
evaluate_mode = True
showcase_mode = False # whether to plot the loss/accuracy over batches


# Ramp up/stabilize cycle sampling parameters
start_prob: float = 0
end_prob: float = 0.1
ramp_epochs: int = 20

# Linear sampling parameters
increase_rate: float = 0.01
limit: float = 0.5
warmup_epochs: int = 100

if training_mode:
    history_callback = HistoryCallback(100)


    scheduled_sampling_callback = ScheduledSamplingCallback(
        start_prob=start_prob,
        end_prob=end_prob,
        ramp_epochs=ramp_epochs,
    )

    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=final_epoch,
        initial_epoch=init_epoch,
        callbacks=[
            history_callback,
            scheduled_sampling_callback,
  k          tf.keras.callbacks.TerminateOnNaN(),
        ],
        verbose=0,
    )

    model.save(SAVE_PATH)
    print(f"Model saved successfully to {SAVE_PATH}\n")
    history_callback.save_to_json(history_save_location)
else:
    print("Model is not in training mode.")
