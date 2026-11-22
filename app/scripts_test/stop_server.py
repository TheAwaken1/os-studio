import psutil
found = False
for c in psutil.net_connections('tcp'):
    if c.laddr and c.laddr.port == 7861 and c.status == 'LISTEN':
        found = True
        p = psutil.Process(c.pid); fam = p.children(recursive=True) + [p]
        par = p.parent()
        if par and 'app.py' in ' '.join(par.cmdline()): fam.append(par)
        for q in fam:
            try: q.kill(); print('killed', q.pid)
            except Exception as e: print('skip', q.pid, e)
print('stopped' if found else 'nothing was listening on 7861')
