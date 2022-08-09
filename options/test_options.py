from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./testresults/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        self.isTrain = False
        return parser
