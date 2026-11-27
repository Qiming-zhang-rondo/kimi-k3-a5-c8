from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"framework_integration"))
from sfa_recent128.producer_reference import expected_ring_writes


class ProducerReferenceTests(unittest.TestCase):
    def call(self,**changes):
        values=dict(token_ring_slots=[0,0,0,0,7,7,7,7],
            token_positions=[254,255,256,257]*2,token_request_epochs=[3]*4+[9]*4,
            token_valid=[True]*8,request_pool_size=8,ring_capacity=256,max_query_per_request=4)
        values.update(changes)
        return expected_ring_writes(**values)

    def test_pool_and_position_are_independent_and_wrap_without_cross_request_alias(self):
        result=self.call()
        self.assertEqual(result["destination_rows"],[254,255,0,1,2046,2047,1792,1793])
        self.assertEqual(result["position_tags"],[254,255,256,257]*2)
        self.assertEqual(result["epoch_tags"],[3]*4+[9]*4)
        self.assertEqual(result["valid_write_count"],8)
        self.assertNotEqual(result["destination_rows"],[0]*4+[7]*4)

    def test_inactive_padding_is_a_complete_noop(self):
        result=self.call(token_ring_slots=[0,-1],token_positions=[256,-1],
            token_request_epochs=[8,-1],token_valid=[True,False])
        self.assertEqual(result["destination_rows"],[0,None])
        self.assertEqual(result["epoch_tags"],[8,None])

    def test_bad_metadata_capacity_or_multiple_writers_are_rejected(self):
        for changes in (dict(request_pool_size=7),dict(ring_capacity=128),
                dict(token_positions=[-1]*8),dict(token_request_epochs=[-1]*8),
                dict(token_positions=[0]*8),dict(max_query_per_request=3),
                dict(token_valid=[1]*8),dict(token_positions=[0])):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.call(**changes)


if __name__=="__main__":
    unittest.main()
