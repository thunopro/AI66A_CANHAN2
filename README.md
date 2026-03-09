# AI66A_CANHAN2

link latex : https://www.overleaf.com/read/fmnzpwsxvmsb#2de848

# week 1 
# DATA INFORMATIONS 

- attack_logs : thu muc chua nhat ky cua attack_logs  
+ attack_logs_extern : nhat ky cuoc tan cong tu ben ngoai 
+ attack_logs_intern : nhat ky cuoc tan cong ty ben trong 

- client_confs : chua cac file cai dat cac may tinh ao. tao ra de tu dong luot web, gui email 
- client_logs : nhat ky cac hanh dong binh thuong ma cac may tinh ao thuc hien 

- traffic : cac thong so hoat dong 
+ external : du lieu thu thap tu cac may ao ve internet ( ngoai xa hoi )
+ openstack : du lieu thu tap tu tao ra tu cac may ao 


# Each row describe one flow. one flow have srcIP,srcPort,destIP,destPort,Protocol,Data first seen, Duration, Bytes, Packets, Flags, Class. 

Understand Data : 
- srcIP, srcPort,destIp, destPort is normal 
- Protocal have four types is TCP,UDP,ICMP,IGCP 
- Data first seen, Duration Bytes, Packets is normal 
- Flags : is flag of TCP 
- Class is target of problem have five types : normal,attacker,victim,suspicious or unknown.  

problem is 