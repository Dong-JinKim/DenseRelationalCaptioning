import json
output=json.load(open('eval/output.json'))
print 'average METEOR : %f'%(output['average_score']*100)
