
PROMPT_SG_OPEN ='Generate a structured scene graph for an image using the following format:\n\n```json\n{\n  "objects": [\n    {"id": "object_name.number", "bbox": [x1, y1, x2, y2]},\n    ...\n  ],\n  "relationships": [\n    {"subject": "object_name.number", "predicate": "relationship_type", "object": "object_name.number"},\n    ...\n  ]\n}\n```\n\n### **Guidelines:**\n- **Objects:**\n  - Assign a unique ID for each object using the format `"object_name.number"` (e.g., `"person.1"`, `"bike.2"`).\n  - Provide its bounding box `[x1, y1, x2, y2]` in integer pixel format.\n  - Include all visible objects, even if they have no relationships.\n\n- **Relationships:**\n  - Represent interactions accurately using `"subject"`, `"predicate"`, and `"object"`.\n  - Omit relationships for orphan objects.\n\n### **Example Output:**\n```json\n{\n  "objects": [\n    {"id": "person.1", "bbox": [120, 200, 350, 700]},\n    {"id": "bike.2", "bbox": [100, 600, 400, 800]},\n    {"id": "helmet.3", "bbox": [150, 150, 280, 240]},\n    {"id": "tree.4", "bbox": [500, 100, 750, 700]}\n  ],\n  "relationships": [\n    {"subject": "person.1", "predicate": "riding", "object": "bike.2"},\n    {"subject": "person.1", "predicate": "wearing", "object": "helmet.3"}\n  ]\n}\n```\n\nNow, generate the complete scene graph for the provided image:\n'



PROMPT_SG = """Generate a structured scene graph for an image using the following format:
<think> [Describe your reasoning process here] </think><answer>
```json
{
  "objects": [
    {"id": "object_name.number", "bbox": [x1, y1, x2, y2]},
    ...
  ],
  "relationships": [
    {"subject": "object_name.number", "predicate": "relationship_type", "object": "object_name.number"},
    ...
  ]
}
```
</answer>

### **Guidelines:**
- **Objects:**
  - Assign a unique ID for each object using the format `"object_name.number"` (e.g., `"person.1"`, `"bike.2"`).
  - Provide its bounding box `[x1, y1, x2, y2]` in integer pixel format.
  - Include all visible objects, even if they have no relationships.

- **Relationships:**
  - Represent interactions accurately using `"subject"`, `"predicate"`, and `"object"`.
  - Omit relationships for orphan objects.

### **Example Output:**
<think> [Describe your reasoning process here] </think><answer>
```json
{
  "objects": [:
    {"id": "person.1", "bbox": [120, 200, 350, 700]},
    {"id": "bike.2", "bbox": [100, 600, 400, 800]},
    {"id": "helmet.3", "bbox": [150, 150, 280, 240]},
    {"id": "tree.4", "bbox": [500, 100, 750, 700]}
  ],
  "relationships": [
    {"subject": "person.1", "predicate": "riding", "object": "bike.2"},
    {"subject": "person.1", "predicate": "wearing", "object": "helmet.3"}
  ]
}
```
</answer>

Now, generate the complete scene graph for the provided image using the format and tags above:
"""




PROMPT_DET = """Given an image, detect all objects using the following format:
```json
{
 "objects":[
 {"id: "object_name.number", "bbox": [x1, y1, x2, y2]},
 ...
 ]
}
```
### **Guidelines:**
- Assign a unique ID for each object using the format `"object_name.nuber"` (e.g., `"person.1"`, `"cat.2"`).
- Provide its bouding box `[x1, y1, x2, y2]` in integer pixel format.
- Include all visible objects.

### **Example Output:**
```json
{
  "objects":[
    {"id": "person.1", "bbox": [120, 200, 350, 700]},
    {"id": "bike.2", "bbox": [100, 600, 400, 800]},
    {"id": "helmet.3", "bbox": [150, 150, 280, 240]},
    {"id": "tree.4", "bbox": [500, 100, 750, 700]}
 ]
}
```
Now, generate the result for the provided image:
"""


OBJ_HOLDER = "OBJECTS_HOLDER"

PROMPT_CLS = """Given an image and a set of detected objects ```OBJECTS_HOLDER```, identify all possible visual relationships between two objects.
### **Guidelines:**
- **Objects:**
  - Objects are formatted as `"object_name.number"` (e.g., `"person.1"`, `"cat.2"`).

- **Relationships:**
  - Represent interactions accurately using `"subject"`,  `"predicate"`, and `"object"`.
  - Omit relationships for orphan objects.

### **Example Output:**
```json
{
  "relationships": [
    {"subject": "person.1", "predicate": "riding", "object": "bike.2"},
    {"subject": "person.1", "predicate": "wearing", "object": "helme.3"}
  ]
}
```
Now, generate the result for the provided image:
"""
