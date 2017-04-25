GET /info
^^^^^^^^^

Description
    Detailed information about the Instaseis database offered from this
    particular server.

Content-Type
    application/json; charset=UTF-8

Example Response
    .. code-block:: json

        {
            "attenuation": true,
            "axisem_version": "615a180",
            "compiler": "ifort 1400",
            "components": "vertical and horizontal",
            "datetime": "2014-11-07T18:48:29.000000Z",
            "directory": "",
            "dt": 0.4874457469080638,
            "dump_type": "displ_only",
            "excitation_type": "dipole",
            "filesize": 948145144202,
            "format_version": 7,
            "is_reciprocal": true,
            "length": 3699.713219032204,
            "max_d": 180,
            "max_radius": 6371,
            "min_d": 0,
            "min_radius": 5671,
            "nfft": 16384,
            "npts": 7591,
            "period": 2,
            "planet_radius": 6371000,
            "slip": [ "..." ],
            "sliprate": [ "..." ],
            "sampling_rate": 2.05151036057477,
            "source_depth": null,
            "spatial_order": 4,
            "src_shift": 3.4121203422546387,
            "src_shift_samples": 7,
            "stf": "errorf",
            "time_scheme": "symplec4",
            "user": "di29kub on login05",
            "velocity_model": "ak135f",
            "external_model_name": ""
        }
