poetry:
C://Users/knerb/AppData/Roaming/Python/Scripts/poetry
Der Run war gut: FireboyAndWatergirl-sac-v0__sac_atari__2__1745589629
Der RUn war somewhat repaired: 
- FireboyAndWatergirl-sac-v0__sac_atari__1__1746427057
- FireboyAndWatergirl-sac-v0__sac_atari__1__1746429815
- FireboyAndWatergirl-sac-v0__sac_atari__1__1746449297
- FireboyAndWatergirl-sac-v0__sac_atari__1__1746452582 excellent
  - game doesnt restart when player dies
  - learning progress starts ~22.5k and rises slowly
PPO lernt:
  - FireboyAndWatergirl-ppo-v0____1__1746603267
  - aber die Länge des Experiments beeinflusst die Performance
DQN lernt:
  - FireboyAndWatergirl-dqn-v0____1__1746614374
  - aber nicht so gut wie PPO, vllt weil Catastrophic forgetting?
Mittwoch Fazit:
- nicht ins Feuer/Wasser zu laufen kann gelernt werden
- aber Exploration ist sehr schwierig
  - mehr Zeit hilft nicht
  - mehr ENvs helfen nicht
  - Hyperparametertuning hilft nicht
- ich muss für jeden Schritt eine Belohnung geben
- Warum nicht einfach auf jedes "Luft"-TIle einen alternierenden Stern
  --> visitedBlocks, irgendwann Belohnung für leere Tiles -> fürs Erkunden, 
- ppo: huge performance increase through Async
  -  27min für 4Mio bei 16 ENvs
  -  23min für 4Mio bei 32 ENvs