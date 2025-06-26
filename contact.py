import tqdm


def get_contact_area(check_watertight: bool, extended_femur_meshes, extended_tibia_meshes):
    if not check_watertight:
        return None
    return [fm.intersection(tm) for fm, tm in tqdm.tqdm(zip(extended_femur_meshes, extended_tibia_meshes))]


def get_contact_components(contact_areas):
    if contact_areas is None:
        return None
    res = []
    for contact_area in contact_areas:
        res.append(contact_area.split())
    return res
