from lstm_model import LSTMModel
if __name__ == "__main__":
    lstm_model = LSTMModel.create_from_args(is_sample_mode=True)
    lstm_model.run()