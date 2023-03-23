echo "%   user"
echo "============"

# collect the data
for user in `ps ax o user:16,pid,pcpu,pmem,vsz,rss,stat,start_time,time,cmd | grep -v COMMAND | awk '{print $1}' | sort -u`
do
  stats="$stats\n`ps ax o user:16,pid,pcpu,pmem,vsz,rss,stat,start_time,time,cmd | egrep ^$user | awk 'BEGIN{total=0}; \
    {total += $4};END{print total,$1}'`"
done

# sort data numerically (largest first)
echo -e $stats | grep -v ^$ | sort -rn | head
