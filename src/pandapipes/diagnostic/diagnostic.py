from pandapipes.diagnostic.diagnostic_functions import (
    default_diagnostic_functions,
    default_argument_values,
)

import logging
import warnings

logger = logging.getLogger(__name__)

log_format_len = 60
log_message_sep = f"\n{'':-<{log_format_len}}\n"


class Diagnostic:

    def __init__(self, add_default_functions=True):
        self._functions = []
        self.kwargs = {}
        self.diag_results = {}
        self.diag_errors = {}
        self.net = None

        if add_default_functions:
            self._functions = default_diagnostic_functions
            self.kwargs = default_argument_values.copy()

    def register_function(self, diagnostic_function, argument_names=None, name=None):
        if name is None:
            name = diagnostic_function.__class__.__name__

        self._functions.append(
            (name, diagnostic_function, argument_names)
        )

    def diagnose_network(self, net, report=True, return_result_dict=True, **kwargs):
        self.diag_results = {}
        self.diag_errors = {}
        self.net = net

        self.kwargs.update(kwargs)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for name, check, argument_names in self._functions:
                if argument_names is None:
                    args = self.kwargs
                else:
                    args = {}
                    for arg_name in argument_names:
                        args[arg_name] = self.kwargs[arg_name]

                try:
                    result = check.diagnostic(net, **args)

                    if result is not None:
                        self.diag_results[name] = result

                except Exception as e:
                    self.diag_errors[name] = e

        if report:
            self.report()

        if return_result_dict:
            return self.diag_results

        return None

    def report(self):
        if self.net is None:
            raise RuntimeError(
                "Network has not been diagnosed yet. Call 'diagnose_network' first."
            )

        original_level = logger.getEffectiveLevel()
        logger.setLevel(logging.INFO)

        logger.warning(
            f"\n\n{' PANDAPIPES DIAGNOSTIC TOOL ':-^{log_format_len}}\n"
        )

        printed_anything = False

        for name, check, _ in self._functions:
            has_result = (
                name in self.diag_results
                or name in self.diag_errors
            )

            if not has_result:
                continue

            if printed_anything:
                logger.warning(log_message_sep)

            check.report(
                self.diag_errors.get(name, None),
                self.diag_results.get(name, None)
            )

            printed_anything = True

        logger.warning(
            f"\n\n{' END OF PANDAPIPES DIAGNOSTIC ':-^{log_format_len}}\n"
        )

        logger.setLevel(original_level)