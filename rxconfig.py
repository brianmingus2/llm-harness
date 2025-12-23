import reflex as rx

config = rx.Config(
    app_name="harness",
    app_module_import="harness",
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
    state_auto_setters=True,
    loglevel=rx.constants.LogLevel.CRITICAL,
)
