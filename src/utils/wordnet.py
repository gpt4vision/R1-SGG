import nltk
from nltk.corpus import wordnet
import json

# Ensure nltk resources are downloaded
#nltk.download('wordnet')

def find_synonym_map(set_a, set_b_list):
    """
    Builds a synonym map for items in set A if a synonym exists in set B.

    Args:
        set_a: A list of strings (set A).
        set_b_list: A list of strings (set B).

    Returns:
        A dictionary representing the synonym map.
        Keys are items from set A that have synonyms in set B.
        Values are the corresponding synonym items from set B.
    """
    synonym_map = {}
    set_b = set(set_b_list) # Convert set B list to set for faster lookup

    for item_a in set_a:
        found_synonym = False
        # First, check for direct match (item_a itself in set_b)
        if item_a in set_b:
            synonym_map[item_a] = item_a
            found_synonym = True
        else:
            # If no direct match, try to find synonyms using WordNet
            for syn in wordnet.synsets(item_a):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name()
                    if lemma_name in set_b:
                        synonym_map[item_a] = lemma_name
                        found_synonym = True
                        break # Found a synonym, move to next item in set_a
                if found_synonym:
                    break # Found a synonym, move to next item in set_a
        if not found_synonym:
            # Check for synonyms by splitting words if item_a is a phrase
            words_in_a = item_a.split()
            if len(words_in_a) > 1:
                for word_a in words_in_a:
                    if word_a in set_b:
                        synonym_map[item_a] = word_a
                        found_synonym = True
                        break
                    else:
                        for syn in wordnet.synsets(word_a):
                            for lemma in syn.lemmas():
                                lemma_name = lemma.name()
                                if lemma_name in set_b:
                                    synonym_map[item_a] = lemma_name
                                    found_synonym = True
                                    break
                            if found_synonym:
                                break
                    if found_synonym:
                        break


    return synonym_map



if __name__ == "__main__":
    set_A = ['quilt', 'sensor', 'fish', 'wall socket', 'goat', 'mouse', 'snow globe', 'santa', 'tie', 'mushroom', 'storefront', 'leaf', 'jar', 'feeder', 'child', 'crib', 'propeller', 'headboard', 'outhouse', 'lemon', 'weather vane', 'snowboarder', 'bike route', 'statue', 'cork', 'ice rink', 'log cabin', 'donkey', 'catering truck', 'flowerbed', 'boarding bridge', 'post it', 'lid', 'saucer', 'water tower', 'blue pants', 'hand', 'planter', 'brickwall', 'red bandana', 'bowl', 'banana', 'wheat thins', 'keyboard', 'wine bottle', 'crosswalk', 'game machine', 'sandals', 'mongoose', 'apple', 'pond', 'clothes', 'marker', 'white chair', 'earphones', 'medals', 'blocks', 'panda', 'saw', 'sun', 'taxi', 'blue mug', 'rugby ball', 'first aid kit', 'candle', 'kite', 'zebra', 'pumpkin', 'yellow truck', 'hotdog', 'bookstore', 'charger', 'grape vine', 'file', 'speaker', 'wetsuit', 'portrait', 'gondola', 'coca cola', 'leaves', 'purple flowers', 'pancake', 'racket', 'birdhouse', 'tablecloth', 'chicken', 'sticker', 'ceiling fan', 'scaffolding', 'sunglasses', 'cheese', 'aircraft carrier', 'sleeping bag', 'rock', 'telescope', 'boat', 'piano', 'flagpole', 'brick wall', 'coleslaw', 'tablet', 'shoes', 'thermos', 'corkboard', 'cowboy hat', 'cheerios', 'restaurant sign', 'floral pattern', 'sand dune', 'mug', 'buildings', 'oar', 'towel holder', 'toaster', 'backpack', 'zebra bag', 'crane', 'vase', 'server rack', 'wood', 'beer glass', 'tv stand', 'can', 'moon', 'fireplug', 'pen holder', 'antelope', 'keys', 'whipped cream', 'shelves', 'external drive', 'driver', 'calendar', 'bathroom', 'church', 'white truck', 'plate', 'vegetables', 'book', 'lights', 'hair dryer', 'knife block', 'hill', 'glue', 'star', 'stuffed animal', 'countertop', 'rainbow', 'tracks', 'green beans', 'noodles', 'chips', 'coral', 'wine', 'pepsi', 'jacket', 'police', 'bridge', 'school bus', 'slide', 'ball', 'helicopter', 'cymbal', 'pot', 'porta potty', 'wave', 'refrigerator', 'arch', 'ribs', 'basketball', 'dog', 'clouds', 'projector', 'sandbag wall', 'glasses', 'pillars', 'knee pads', 'folder', 'palm', 'bread', 'tent', 'napkin', 'gazebo', 'printer', 'peas', 'sauce', 'police car', 'lamp shade', 'picnic table', 'cigarette', 'staircase', 'driveway', 'yellow mug', 'candy', 'mccafe drink', 'soccer ball', 'coffee pot', 'mousepad', 'snow sign', 'fence', 'crates', 'well', 'asparagus', 'go sign', 'pail', 'stapler', 'bookcase', 'field', 'surfer', 'vending machine', 'store', 'buoy', 'postbox', 'tank', 'floor', 'internet', 'wardrobe', 'lake', 'water', 'hammer', 'sheep', 'pancakes', 'sky', 'paint', 'wreath', 'duck', 'wooden box', 'paintbrush', 'bouquet', 'jars', 'tray', 'pipe', 'tissue box', 'net', 'engine', 'closet', 'salad', 'toothpaste', 'vent', 'feathers', 'river', 'tennis bag', 'greenhouse', 'sculpture', 'bed', 'green fabric', 'name card', 'rockwall', 'scorpions', 'path', 'record player', 'parking sign', 'giraffe', 'watch', 'bath', 'coca cola bottle', 'metal grates', 'air conditioner', 'pitcher', 'deer head', 'menu board', 'bun', 'ski poles', 'grass', 'wrench', 'butterfly', 'purse', 'iceberg', 'sidewalk', 'flowers', 'gray car', 'atm', 'ice', 'socks', 'tractor', 'egg', 'dock', 'wind turbine', 'gloves', 'clock', 'kiteboard', 'barn', 'ornament', 'shell', 'toilet tissue', 'cow', 'truck', 'blinds', 'table', 'podium', 'ship', 'bottle', 'rocks', 'cucumber', 'seat', 'billboard', 'file cabinet', 'ruler', 'chain', 'dough', 'meat', 'scooter', 'green fridge', 'wind chime', 'microphone', 'tomato', 'ski lift', 'chocolate sauce', 'shelf', 'fries', 'oven', 'reflection', 'power pole', 'cross', 'cutlery holder', 'directory', 'berries', 'water glass', 'couscous', 'ottoman', 'box', 'plant', 'water bottle', 'snowboard', 'door', 'pizza', 'waterfall', 'stroller', 'stop sign', 'bus', 'paper bag', 'snack bar', 'headphones', 'balcony', 'bowling lane', 'apples', 'bedroom', 'tennis shorts', 'pineapple', 'stained glass', 'canvas', 'milk box', 'cutting board', 'shed', 'cereal', 'van', 'basket', 'toothbrush', 'tricycle', 'basketball court', 'fire extinguisher', 'pants', 'grill', 'boxcar', 'rug', 'greenwall', 'exit sign', 'conveyor', 'house', 'flower hanging', 'fire hydrant', 'bowling lanes', 'barrel', 'blanket', 'tile', 'cake', 'tv', 'streetlight', 'laptop', 'stone wall', 'water heater', 'sand', 'yellow strap', 'hamburger', 'game case', 'pole', 'tower', 'minnie', 'vehicle', 'firetruck', 'spectators', 'mirror', 'fridge', 'books', 'sea lion', 'tree', 'scoreboard', 'bulletin board', 'file drawer', 'board', 'hat', 'wheelchair', 'tennis player', 'wall', 'fountain', 'candles', 'ceiling', 'ambulance', 'palm tree', 'beach', 'window', 'building', 'police officer', 'drill', 'bar', 'remote', 'coffee table', 'crosswalk sign', 'lightswitch', 'jeans', 'basketball jersey', 'sandbox', 'gun', 'kiwi', 'microwave', 'power line', 'tail', 'no biking sign', 'lamp base', 'restaurant', 'stairs', 'sofa', 'pool', 'bike lane sign', 'wii', 'home plate', 'antenna', 'parrot', 'gravel', 'elephant', 'iguana', 'tag', 'bacon', 'toilet', 'squirrel', 'hot tub', 't shirt', 'side table', 'street', 'bat', 'doll', 'whiteboard', 'potatoes', 'label', 'pool table', 'sink', 'pear', 'coffee', 'skis', 'gas pump', 'placemat', 'whirlpool', 'sugar pack', 'stool', 'paper', 'frisbee', 'monitor', 'car', 'handbag', 'courtroom', 'shower', 'snowflake', 'castle', 'wine glass', 'track', 'number', 'wagon', 'ocean', 'glass roof', 'cauliflower', 'gate', 'parking by permit only', 'concrete path', 'glass', 'shirt', 'bike rack', 'plastic container', 'stove', 'paper towel', 'cup', 'snow', 'container', 'toys', 'fan', 'base', 'bowling ball', 'stuffed toy', 'lime', 'parking permit only sign', 'ski', 'pen', 'person', 'rocky slope', 'carrousel', 'curtain', 'laptop case', 'towel', 'sandwich bar', 'baby', 'water fountain', 'shrub', 'cones', 'chair', 'lottery sign', 'baptism tub', 'railing', 'goggles', 'fork', 'green car', 'forklift', 'road', 'cushion', 'cabinet', 'computer', 'juice', 'camera', 'lock', 'power lines', 'high heel shoe', 'red car', 'bowling pins', 'blue seat', 'bed sheet', 'doily', 'vacuum', 'ice cream', 'coffee cup', 'phone booth', 'bucket', 'ping pong table', 'phone', 'violin', 'lampshade', 'rice', 'cloud', 'jet', 'structure', 'sailboat', 'green plants', 'houses', 'invitation', 'pine tree', 'speedometer', 'skier', 'lamp', 'belt', 'text', 'concrete wall', 'green shirt', 'white gate', 'tarp', 'foosball table', 'desk', 'carrots', 'wafer', 'busstop', 'surfboard', 'couch', 'blue court', 'grapes', 'trolley', 'pouch', 'television', 'lighthouse', 'leg', 'soda can', 'candy bar', 'barbed wire', 'sweater', 'fish tank', 'bell', 'bookshelf', 'sack', 'seats', 'bison', 'chocolate', 'guardrail', 'pitcher mound', 'cactus', 'flipflops', 'cross traffic sign', 'scissors', 'hoop', 'airplane', 'carrot', 'underwear', 'food truck', 'bus sign', 'raft', 'bulldozer', 'dishwasher', 'fruit', 'map', 'mat', 'sail', 'drum', 'lanyard', 'toy', 'painting', 'cups', 'ground', 'pump', 'branch', 'spectator', 'inflatable', 'vegetation', 'stage', 'yellow flower', 'concrete', 'green box', 'button', 'food', 'parking meter', 'log', 'cable', 'scrambled eggs', 'clock tower', 'blue wall', 'motorcycle', 'washing machine', 'anchor', 'cockpit', 'screwdriver', 'shutters', 'tennis court', 'baby monkey', 'strawberry', 'skeleton', 'wing', 'train track', 'lollipop', 'balloon', 'tennis racket', 'candle holder', 'horse', 'goal', 'bath mat', 'shoe', 'dryer', 'baseball field', 'flag', 'beer bottle', 'chocolate dessert', 'lunchbox', 'milking machine', 'trashcan', 'ski pole', 'garage', 'skull', 'cd', 'shutter', 'knife', 'cell phone', 'wheel', 'manhole', 'roof', 'counter', 'battery', 'people', 'soil', 'broccoli', 'white cabinet', 'teddy bear', 'tissue paper', 'drain', 'banana leaf', 'papers', 'coffee maker', 'pufferfish', 'z crossing', 'fire', 'sign', 'guitar', 'chimney', 'cheetah', 'kettle', 'corn', 'bathtub', 'bag', 'sausage', 'train', 'fireplace', 'cloth', 'paella', 'platform', 'ramp', 'fire escape', 'pepper', 'mountain', '4 way sign', 'hedge', 'toast', 'calculator', 'trash can', 'boy', 'menu', 'sandwich', 'bank', 'salt', 'lifeguard', 'mountains', 'mobile home', 'frying egg', 'street sign', 'spaghetti', 'magazine', 'nest', 'water meter', 'goalpost', 'yellow pole', 'donut', 'ladder', 'drawer', 'shorts', 'escalator', 'newspaper', 'toilet paper', 'store sign', 'plastic bag', 'tennis net', 'street light', 'monkey bar', 'parking lot', 'telephone pole', 'case', 'spatula', 'soap', 'pluto', 'scarf', 'brick column', 'screen', 'bush', 'motor', 'elevator', 'microphone stand', 'blue building', 'tennis ball', 'spoon', 'flower arrangement', 'computer monitor', 'bench', 'dirt', 'pillow', 'woman', 'lettuce', 'dresser', 'ketchup', 'tea', 'metal object', 'tow truck', 'light', 'mouse pad', 'potted plant', 'bus stop', 'no parking sign', 'fruits', 'moss', 'plug', 'thermostat', 'traffic light', 'machine', 'goose', 'tissue', 'polling station sign', 'earrings', 'mirror ball', 'umbrella', 'fishbowl', 'nightstand', 'baseball', 'mailbox', 'towel bar', 'trees', 'scorpion', 'cd rack', 'glove', 'office', 'picture frame', 'white fridge', 'notebook', 'flower', 'knee pad', 'poster', 'handle', 'faucet', 'minibar', 'plane', 'bicycle', 'spiderman', 'column', 'soap dish', 'kitkat', 'carriage', 'hose', 'frame', 'potato', 'dome', 'cereal box', 'spray bottle', 'trash', 'trailer', 'sidecar', 'chandelier', 'collaborate sign', 'tripod', 'record', 'garbage can', 'monkey', 'watering can', 'street lamp', 'cat', 'mannequin', 'helmet', 'baseball glove', 'socket', 'valve', 'cowboy', 'arm', 'straw', 'toilet seat', 'wallet', 'collar', 'bird', 'package', 'grape', 'bear', 'cone', 'penne', 'phonebooth', 'no entry', 'radiator', 'lily pad', 'one way sign', 'bagel', 'lantern', 'armchair', 'air hockey table', 'orange juice', 'stripes', 'suv', 'suitcase', 'bike', 'alarm', 'steak', 'grate', 'toilet brush', 'luggage', 'onions', 'pier', 'skateboard', 'cash register', 'spire', 'pan', 'red counter', 'picture', 'orange', 'teapot', 'tank top', 'cart', 'rope', 'man', 'windmill', 'mashed potatoes', 'eraser', 'yogurt', 'tiger']
    set_B_list = ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']


    print(find_synonym_map(["bicycle"], set_B_list))
    import pdb; pdb.set_trace()

    synonym_mapping = find_synonym_map(set_A, set_B_list)
    print(synonym_mapping)
    with open("synonym_mapping.json", 'w') as fout:
        json.dump(synonym_mapping, fout)
    
