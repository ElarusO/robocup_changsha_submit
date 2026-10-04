from entry_base import build_policy as build_base_policy
from h3_pursuit_policy import PursuitReference

def build_policy(context):
    return PursuitReference(build_base_policy(context))
