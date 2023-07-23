from .preresnet import *
from .vgg import *
from .vggleaky import *
from .vggtest import *
from .vgg2 import *
from .wide_resnet import *
from .resnet import *
from .vgg2 import *
from .vgg_drop import *

#Added by  24/03/2020
from .MLP import *
from .lenet import *

# Added by 20 Oct
from .resnext2 import *
from .densenet import *

# Added by 2 Dec
from .all_cnn import *

# Added by  on 23 Dec - the RNN/LSTM model for NLP tasks
#from .language_model import *

# Added by  on 26 Dec - logistic regression toy example
from .logistic_regression import *

# Added by  on 1 Jan - added Shake-shake architecture
from .shakeshake import *

import torchvision.models as modelstorch
resnet50 = modelstorch.resnet50(pretrained=False)
resnet18 = modelstorch.resnet18()
