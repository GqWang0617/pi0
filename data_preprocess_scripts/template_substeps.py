# template_substeps = dict(  # fold one shirt new version
#     standard_substeps=['Grasp the left hem and left sleeve, then fold them forward.',
#                        'Pull back to flatten the fabric.',
#                        'Grasp the right hem and right sleeve, then fold them backward.',
#                        'Fold to the left.',
#                        'Wrap to the right.',
#                        'Move the folded cloth to right.'],
#     recover_steps=['Grasp failed, please try again.',
#                    'The hem is turned up, Grasp it to straighten it.',
#                    ]
# )
# template_substeps = dict( # old version
#     standard_substeps=['Grasp the left hem and left sleeve, then fold them forward.',
#         'Grasp the hem and collar, then pull back to flatten the fabric.',
#         'Grasp the right hem and right sleeve, then fold them backward.',
#         'Grasp the collar and fold it to the left.',
#         'Grasp the middle of hem and fold it to the right.',
#         'Grasp the collar and move it to the right.'],
#     recover_steps=['Grasp failed, please try again.',
#         'The hem is turned up, Grasp it to straighten it.',
#         ]
# )

# template_substeps = dict( # fold one shirt and stack
#     standard_substeps=['Grasp the left hem and left sleeve, then fold them forward.',
#         'Pull back to flatten the fabric.',
#         'Grasp the right hem and right sleeve, then fold them backward.',
#         'Fold to the left.',
#         'Wrap to the right.',
#         'Grab the folded cloth and move back.',
#         'Stack the clothes.',
#         'Move the folded clothes to right.',
#         ],
#     recover_steps=['Grasp failed, please try again.',
#         'The hem is turned up, Grasp it to straighten it.',
#         ]
# )

# template_substeps = dict( # fold one random shirt and stack, random on the left
#     standard_substeps=['Drag the t-shirt to the middle of table.',
#         'Grasp the left hem then find the left sleeve. Then fold them forward.',
#         'Pull back to flatten the fabric.',
#         'Grasp the right hem and right sleeve, then fold them backward.',
#         'Fold to the left.',
#         'Wrap to the right.',
#         'Grab the folded cloth and move back.',
#         'Stack the clothes.',
#         'Move the folded clothes to right.',
#         ],
#     recover_steps=['Grasp failed, please try again.',
#         'The hem is turned up, Grasp it to straighten it.',
#         ]
# )

# template_substeps = dict(  # fold one random shirt from basket and stack
#     standard_substeps=['Grab the t-shirt from the basket and place it on the table.',
#                        'Grasp the left hem then find the left sleeve. Then fold them forward.',
#                        'Pull back to flatten the fabric.',
#                        'Grasp the right hem and right sleeve, then fold them backward.',
#                        'Fold to the left.',
#                        'Wrap to the right.',
#                        'Grab the folded cloth and move back.',
#                        'Stack the clothes.',
#                        'Move the folded clothes to right.',
#                        ],
#     recover_steps=['Grasp failed, please try again.',
#                    'The hem is turned up, Grasp it to straighten it.',
#                    ]
# )
"""v1: New template of random folding shirts from basket"""
# template_substeps = dict(  # fold one random shirt from basket and stack
#     raw_language='The crumpled shirts are in the basket. Pick it out and fold it.',
#     standard_substeps=['Grab the t-shirt from the basket and place it on the table.',
#                        'Find left hem and left sleeve. Then lift up and fold them forward.',
#                        'Pull back to flatten the fabric.',
#                        'Grasp the right hem and right sleeve, then fold them backward.',
#                        'Fold to the left.',
#                        'Wrap to the right.',
#                        'Grab the folded cloth and move back.',
#                        'Stack the clothes.',
#                        'Move the folded clothes to right.',
#                        ],
#     recover_steps=['Grasp failed, please try again.',
#                    'The hem is turned up, Grasp it to straighten it.',
#                    ]
# )

"""v2: New template of random folding shirts from basket. More fine-grained substeps"""
template_substeps = dict(  # fold one random shirt from basket and stack
    raw_language='The crumpled shirts are in the basket. Pick it out and fold it.',
    standard_substeps=[
        'Grab the t-shirt from the basket and place it on the table.',
       'Find left hem and left sleeve. Straighten the cloth.', # 100
        ## recurrent_steps
       "Fold them forward.",
       'Pull back to flatten the fabric.',
       'Grasp the right hem and right sleeve, then fold them backward.',
       'Fold to the left.',
       'Wrap to the right.',
        ## stack_steps or push_steps
    ],
    stack_steps=[
        'Grab the folded cloth and move back.',
        'Stack the clothes.',
        'Move the folded clothes to right.'
    ],
    push_steps=['Move the folded clothes to right.'],
    recurrent_step=['Lift up the cloth and check it.']
)

# template_substeps = dict( # fold two random shirt from basket and stack
#     standard_substeps=[
#         'Grab the t-shirt from the basket and place it on the table.',
#         'Grasp the left hem then find the left sleeve. Then fold them forward.',
#         'Pull back to flatten the fabric.',
#         'Grasp the right hem and right sleeve, then fold them backward.',
#         'Fold to the left.',
#         'Wrap to the right.',
#         'Move the folded cloth to right.',
#         'Grab the t-shirt from the basket and place it on the table.',
#         'Grasp the left hem then find the left sleeve. Then fold them forward.',
#         'Pull back to flatten the fabric.',
#         'Grasp the right hem and right sleeve, then fold them backward.',
#         'Fold to the left.',
#         'Wrap to the right.',
#         'Grab the folded cloth and move back.',
#         'Stack the clothes.',
#         'Move the folded clothes to right.',
#         ],
#     recover_steps=['Grasp failed, please try again.',
#         'The hem is turned up, Grasp it to straighten it.',
#         ]
# )