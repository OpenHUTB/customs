import commands
a,b = commands.getstatusoutput('eval.sh')
print(b)
