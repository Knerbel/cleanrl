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
-  good run: FireboyAndWatergirl-dqn-v0____1__4000000__0.00025__10000__0.99__1.0__500__128__1__0.05__0.5__100000__10__1747030630 recurring reward
-  single reward -> less learning
ALLES bis zum 13.05 hat nur bedingt gelernt
- Grundlegend etwas falsch?
  - Reward function eigentlich vielversprechend
  - Observation_space eigentlich vielversprechend
  - ABER: z.B. PPO verwendet NN und kein CNN -> besser für TabData und weniger für Bilder! -> Kapitel in MT
- Deswegen: Experimente mit CNN statt NN
  - erster Run: FireboyAndWatergirl-ppo-v0__ppo_atari__1__1747201908
    - kein wirkliches lernen, recurring, 90/10
  - zweiter RUn: FireboyAndWatergirl-ppo-v0__ppo_atari__1__1747202501
    -  kein wirkliches lernen, single, 90/10
 - Testrun mit "BreakoutNoFrameskip-v4" funktioniert. Aber man muss lange warten BreakoutNoFrameskip-v4__ppo_atari__1__1747208554
   - Im .md video werden die Wrapper erklärt, das muss mit in die MT
- PPO-ATARI erfolgreicher Run: FireboyAndWatergirl-ppo-v3__ppo_atari__1__1747210222
  - ABER mit "nur" 400 steps, und einfachem neuen Level1b und ohne Doors
- PPO-Atari erfolgreicher Run FireboyAndWatergirl-ppo-v3__ppo_atari__1__1747217226
    -  und einfachem neuen Level1c u
NACHDEM ALLES NICHT WIRKLICH GEKLAPPT HAT, neuer Versuch:
- inspiriert von Snake
  - low total_timesteps
  - high num_envs