import torch
import torch.nn as nn


class Forecaster(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=2):
        super(Forecaster, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        h_0 = torch.zeros(1, input.size(0), self.hidden_size).cuda()
        c_0 = torch.zeros(1, input.size(0), self.hidden_size).cuda()

        # h_0 = torch.zeros(1, input.size(0), self.hidden_size)
        # c_0 = torch.zeros(1, input.size(0), self.hidden_size)

        input = input.transpose(0, 1)

        output, (h_n, c_n) = self.rnn(input, (h_0, c_0))

        # print('input', input.size())
        # print('output[-1]', output[-1].size())
        # print()

        output = self.fc(output[-1])
        return output


# class CellForecaster(nn.Module):
#     def __init__(self, input_size=2, hidden_size=128, output_size=2, future=0):
#         super(CellForecaster, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.future = future
#
#         self.rnn = nn.LSTMCell(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size
#         )
#         self.fc = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(self.hidden_size, self.output_size)
#         )
#
#     def forward(self, input):
#         h_t = torch.zeros(input.size(0), self.hidden_size).cuda()
#         c_t = torch.zeros(input.size(0), self.hidden_size).cuda()
#
#         # h_t = torch.zeros(input.size(0), self.hidden_size)
#         # c_t = torch.zeros(input.size(0), self.hidden_size)
#
#         input = input.transpose(0, 1)
#
#         for i in range(input.size(0)):
#             h_t, c_t = self.rnn(input[i], (h_t, c_t))
#
#         for i in range(self.future):
#             h_t, c_t = self.rnn(self.fc(h_t), (h_t, c_t))
#
#         output = self.fc(h_t)
#
#         return output


class CellForecaster(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=2, future=0):
        super(CellForecaster, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.future = future

        self.rnn = nn.LSTMCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, input):
        h_t = torch.zeros(input.size(0), self.hidden_size).cuda()
        c_t = torch.zeros(input.size(0), self.hidden_size).cuda()

        # h_t = torch.zeros(input.size(0), self.hidden_size)
        # c_t = torch.zeros(input.size(0), self.hidden_size)

        input = input.transpose(0, 1)

        for i in range(input.size(0)):
            h_t, c_t = self.rnn(input[i], (h_t, c_t))

        outputs = list()

        for i in range(self.future):
            output = self.fc(h_t)
            outputs.append(output)
            h_t, c_t = self.rnn(self.fc(h_t), (h_t, c_t))

        output = torch.cat(outputs, dim=1)
        # output = self.fc(h_t)

        return output


if __name__ == '__main__':
    # model = Forecaster(input_size=2, hidden_size=128, output_size=2)
    model = CellForecaster(input_size=2, hidden_size=1024, output_size=2, future=12)

    input = torch.randn(2, 12, 2)

    output = model(input)

    print(output.size())

