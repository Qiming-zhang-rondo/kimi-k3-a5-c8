"""Offline addressing oracle for producer tests; never call in a graph path.

Public test logic only, with no vendor source or device-framework imports.
token_ring_slots are request-pool IDs, NOT logical or physical token positions.
"""
import operator


def _integer(value,name,minimum=0):
    if isinstance(value,bool):
        raise ValueError(name+" must be an integer, not bool")
    try:
        result=operator.index(value)
    except TypeError as error:
        raise ValueError(name+" must be an integer") from error
    if result<minimum:
        raise ValueError(name+" is below its minimum")
    return result


def expected_ring_writes(*,token_ring_slots,token_positions,token_request_epochs,
                         token_valid,request_pool_size,ring_capacity,max_query_per_request):
    """Return expected payload rows and tags for saved CPU metadata.

    Inactive rows are no-ops even if their IDs/positions are padding sentinels.
    Duplicate destinations in one invocation are rejected rather than assigning
    a nondeterministic last writer. A private tiled kernel must implement each
    logical write exactly once, or coordinate its column-sharded producers.
    """
    pools=_integer(request_pool_size,"request_pool_size",1)
    capacity=_integer(ring_capacity,"ring_capacity",1)
    bound=_integer(max_query_per_request,"max_query_per_request",1)
    if capacity%128 or capacity<128+bound-1:
        raise ValueError("ring capacity must be 128-aligned and cover window plus query bound")
    vectors=tuple(list(x) for x in (token_ring_slots,token_positions,token_request_epochs,token_valid))
    if len({len(x) for x in vectors})!=1:
        raise ValueError("token metadata lengths must match")
    rows,positions,epochs,seen=[],[],[],set()
    counts={}
    for bank,pos,epoch,active in zip(*vectors):
        if type(active) is not bool:
            raise ValueError("token_valid must contain explicit bool values")
        if not active:
            rows.append(None);positions.append(None);epochs.append(None)
            continue
        bank=_integer(bank,"active request-pool ID")
        pos=_integer(pos,"active logical position")
        epoch=_integer(epoch,"active request epoch")
        if bank>=pools:
            raise ValueError("active request-pool ID is out of range")
        destination=bank*capacity+pos%capacity
        if destination in seen:
            raise ValueError("multiple active tokens target the same ring row")
        seen.add(destination)
        counts[bank]=counts.get(bank,0)+1
        if counts[bank]>bound:
            raise ValueError("active query count exceeds max_query_per_request")
        rows.append(destination);positions.append(pos);epochs.append(epoch)
    return dict(destination_rows=rows,position_tags=positions,epoch_tags=epochs,
        valid_write_count=len(seen),request_pool_size=pools,ring_capacity=capacity,
        formula="bank * capacity + logical_position % capacity",
        scope="CPU reference only; not a kernel implementation or runtime host read")
