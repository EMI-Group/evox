# We import the GPRegression and GPClassification classes from the regression and classification modules
try:
    from .regression import GPRegression
except ImportError as e:
    original_error_msg = str(e)

    def GPRegression(*args, **kwargs):
        raise ImportError(
            f'GPRegression requires gpjax, but got "{original_error_msg}" when importing'
        )


try:
    from .classification import GPClassification
except ImportError as e:
    original_error_msg = str(e)

    def GPClassification(*args, **kwargs):
        raise ImportError(
            f'GPClassification requires gpjax, but got "{original_error_msg}" when importing'
        )
