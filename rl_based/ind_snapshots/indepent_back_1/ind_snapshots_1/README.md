f reduced_enemy > reduced_myself:
                reward = 15
            elif reduced_enemy <= reduced_myself and reduced_enemy > 0:
                reward = 1
            else:
                reward = -10  
