import torch
import os

class BaseNet(torch.nn.Module):
    '''
    Neural net base class.
    Models are saved as '<model_name>_state_dict.pth'.
    '''
    def __init__(self, model_name):
        super().__init__()
        self._model_loaded = False
        self.model_name = model_name
        self.model_path = os.path.join('models', f'{self.name}_state_dict.pth')
        self.device = self._get_device()

    def _get_device(self) -> str:
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f'Device = {device}')
        return device

    def _load_model(self) -> None:
        print(f'Loading {self.name}.pth...')
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            self._model_loaded = True
            print('Model loaded!')
        else:
            print('Model not found!')

    def _save_model(self) -> None:
        if os.path.exists(self.model_path):
            confirm =  input('A model with the same name already exists. Are you sure you want to overwrite? [y/n] ')
            while True:
                if confirm == 'y':
                    torch.save(self.state_dict(), self.model_path)
                    print('Model saved!')
                    break
                elif confirm == 'n':
                    print('Model not saved!')
                    break
                else:
                    confirm = input('Invalid entry. Enter [y/n]: ')
