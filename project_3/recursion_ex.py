
activity_dependencies = {
    'A': '',
    'B': 'A',
    'C': 'B',
    'D': 'C',
    'E': 'C',
    'F': 'E',
    'G': 'D',
    'H': ['E','G'],
    'I': 'C',
    'J': ['F','I'],
    'K': 'J',
    'L': 'J',
    'M': 'H',
    'N': ['K','L']
}

activity_map = {
    'A': 'excavate',
    'B': 'lay_foundation',
    'C': 'rough_wall',
    'D': 'roof',
    'E': 'exterior_plumbing',
    'F': 'interior_plumbing',
    'G': 'exterior_siding',
    'H': 'exterior_painting',
    'I': 'electrical_work',
    'J': 'wallboard',
    'K': 'flooring',
    'L': 'interior_painting',
    'M': 'exterior_fixtures',
    'N': 'interior_fixtures'
}

def get_all_dep(activity):
    dep = activity_dependencies[activity]
    yield dep
    if dep != '':
        if isinstance(dep, str):
            yield from get_all_dep(dep)

        elif isinstance(dep, list):
            for sub_a in dep:
                yield from get_all_dep(sub_a)

prec_dict = {}

for activity in list(activity_dependencies.keys()):
    tasks_forward_set = set(activity_dependencies.keys()).difference(activity)
    
    tasks_forward_list = []
    for dep in get_all_dep(activity):
        tasks_forward_set = tasks_forward_set.difference(dep)

    tasks_forward_list = [activity_map[task] for task in tasks_forward_set]
    prec_dict[activity_map[activity]] = tasks_forward_list