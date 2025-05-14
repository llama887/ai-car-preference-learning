import psutil

def print_system_stats() -> None:
    """
    Print key system statistics:
      • CPU usage
      • RAM usage
      • Swap usage
      • Disk usage (for the root partition)
      • Load average (on Unix)
    """
    # — CPU —
    # Measure CPU utilization over a 1-second sampling window:
    cpu_percent: float = psutil.cpu_percent(interval=1.0)
    print(f"CPU Usage: {cpu_percent:.1f}%")

    # — Memory —
    # virtual_memory() returns a namedtuple with fields:
    #   total, available, percent, used, free, active, inactive, buffers, cached, shared, slab
    vm = psutil.virtual_memory()
    print(f"RAM Total:     {vm.total  / (1024**3):.2f} GB")
    print(f"RAM Available: {vm.available / (1024**3):.2f} GB")
    print(f"RAM Used:      {vm.used     / (1024**3):.2f} GB  ({vm.percent}%)")

    # — Swap —
    # swap_memory() returns: total, used, free, percent, sin, sout
    sw = psutil.swap_memory()
    print(f"Swap Total:    {sw.total / (1024**3):.2f} GB")
    print(f"Swap Used:     {sw.used  / (1024**3):.2f} GB  ({sw.percent}%)")

    # — Disk —
    # disk_usage('/') gives usage stats for the root filesystem
    du = psutil.disk_usage('/')
    print(f"Disk Total:    {du.total / (1024**3):.2f} GB")
    print(f"Disk Used:     {du.used  / (1024**3):.2f} GB  ({du.percent}%)")

    # — Load Average (Unix only) —
    # getloadavg() returns the average system load over 1, 5, and 15 minutes
    try:
        load1, load5, load15 = psutil.getloadavg()
        print(f"Load Avg (1m, 5m, 15m): {load1:.2f}, {load5:.2f}, {load15:.2f}")
    except (AttributeError, OSError):
        # Not available on Windows or some systems
        pass
