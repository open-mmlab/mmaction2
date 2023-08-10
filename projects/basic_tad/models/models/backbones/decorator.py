def crops_to_batch(forward_methods):

    def wrapper(self, inputs, *args, **kwargs):
        num_crops = inputs.shape[1]
        inputs = inputs.view(-1, *inputs.shape[2:])
        return forward_methods(self, inputs, *args, **kwargs)

    return wrapper
