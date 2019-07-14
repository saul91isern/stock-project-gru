import matplotlib.pyplot as plt

class DataView:

    @staticmethod
    def plot_variables(data):
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.plot(data.open.values, color='red', label='open')
        plt.plot(data.adj_close.values, color='green', label='close')
        plt.plot(data.low.values, color='blue', label='low')
        plt.plot(data.high.values, color='black', label='high')
        plt.title('Price')
        plt.xlabel('Time [days]')
        plt.ylabel('Price')
        plt.legend(loc='best')

    @staticmethod
    def plot_model_loss(history):
        plt.figure(figsize=(25, 13))
        plt.subplot(311)
        plt.plot(history.epoch, history.history["loss"])
        plt.plot(history.epoch, history.history["val_loss"])
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.title("Model Behaviour")
        plt.legend(["loss", "val_loss"])

    @staticmethod
    def plot_results_against_true_data(tested_data, true_data):
        plt.figure(figsize = (15,6))
        # style
        plt.style.use('seaborn-darkgrid')
        # create a color palette
        palette = plt.get_cmap('Set1')
        plt.plot(
            range(len(tested_data)), 
            tested_data, 
            marker='', 
            color="blue", 
            linewidth=0.5, 
            alpha=1, 
            label="Predicted Data"
        )

        plt.plot(
            range(len(true_data)), 
            true_data, 
            marker='', 
            color="black", 
            linewidth=0.5, 
            alpha=0.3, 
            label="Real Data"
        )

        plt.legend(loc=2, ncol=2)
        plt.title("Results comparison", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Hours")
        plt.ylabel("Price")
        plt.show()