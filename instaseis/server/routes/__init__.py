import sys

# Necessary to allow event loops in any thread. Default behavior for Python 2.
if sys.version_info.major != 2:  # pragma: no cover
    import asyncio  # NOQA
    import tornado.platform.asyncio  # NOQA

    asyncio.set_event_loop_policy(
        tornado.platform.asyncio.AnyThreadEventLoopPolicy()
    )
