from torch import optim, nn


def get_optimizer(model: nn.Module,
                  learning_rate: float,
                  optimizer_name="ADAM",
                  betas=(0.9, 0.999),
                  eps=1e-8,
                  amsgrad=False,
                  momentum=0,
                  weight_decay=0,
                  dampening=0,
                  nesterov=False,
                  max_iter=20,
                  max_eval=25,
                  tolerance_grad=1e-5,
                  tolerance_change=1e-9,
                  history_size=100,
                  line_search_fn=None):
    if optimizer_name == "ADAM":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               betas=betas,
                               eps=eps,
                               weight_decay=weight_decay,
                               amsgrad=amsgrad)

    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay,
                              dampening=dampening,
                              nesterov=nesterov)

    elif optimizer_name == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(),
                                lr=learning_rate,
                                max_iter=max_iter,
                                max_eval=max_eval,
                                tolerance_grad=tolerance_grad,
                                tolerance_change=tolerance_change,
                                history_size=history_size,
                                line_search_fn=line_search_fn)
    else:
        raise Exception("Not a valid optimizer")

    return optimizer
