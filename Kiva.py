import urllib2
import simplejson as json
from types import *
from datetime import datetime
from urlparse import urljoin
from urllib import urlencode
import re

__version__ = 0.1

API_VERSION = 1

RE_DATE = re.compile('_?date$')
BASE_URL = 'http://api.kivaws.org/v%i/' % API_VERSION
FORMAT = "%Y-%m-%dT%H:%M:%SZ"

SEARCH_STATUS = ['fundraising','funded','in_repayment','paid','defaulted']
SEARCH_GENDER = ['male','female']
SEARCH_REGION = ['na','ca','sa','af','as','me','ee']
SEARCH_SORT   = ['popularity','loan_amount','oldest','expiration',
                 'newest','amount_remaining','repayment_term']

def getRecentLendingActions():
    return __make_call(
        'lending_actions/recent.json', 'lending_actions')

def getLenderInfo(*lender_ids):
    # need one lender, can have up to 50
    if len(lender_ids) == 0:
        raise("Must have at least 1 lender id")
    elif len(lender_ids) > 50:
        raise("Can have up to 50 lender ids; %i submitted" % len(lender_ids))

    lids = ",".join(lender_ids)

    return __make_call('lenders/%s.json' % lids, 'lenders')

def getLenderLoans(lender_id, page=1):
    return __make_call('lenders/%s/loans.json?page=%i' % (lender_id, page),
                       'loans',
                       getLenderLoans, [lender_id])
    
def getNewestLoans(page=1):
    return __make_call('loans/newest.json?page=%i' % page,
                            'loans', getNewestLoans)

def getLoans(*loan_ids):
    if len(loan_ids) == 0 or len(loan_ids) > 10:
        raise("You can request between 1 and 10 loans")
    lids = ",".join(map(lambda x: str(x), loan_ids))
    return __make_call('loans/%s.json' % lids, 'loans')

def getLenders(loan_id, page=1):
    return __make_call('loans/%s/lenders.json?page=%i' % (str(loan_id), page),
                       'lenders', getLenders, [loan_id])

def getJournalEntries(loan_id, include_bulk=True, page=1):
    ib = include_bulk and 1 or 0
    return __make_call('loans/%s/journal_entries.json?page=%i&include_bulk=%s' % (str(loan_id), page, ib),
                       'journal_entries', getJournalEntries, [loan_id, include_bulk])

def getEntryComments(entry_id, page=1):
    return __make_call('journal_entries/%s/comments.json?page=%i' % (str(entry_id), page),
                       'comments', getEntryComments, [entry_id])

def searchLoans(status=None, gender=None, sector=None, region=None,
                country_code=None, partner=None, q=None,
                sort_by=None, page=1):
    opts = {'status':status, 'gender':gender, 'sector':sector,
            'region':region, 'country_code':country_code,
            'partner':partner, 'q':q, 'sort_by': sort_by}
    
    # check params
    status       = __check_param(status,       'status', SEARCH_STATUS)
    gender       = __check_param(gender,       'gender', SEARCH_GENDER)
    sector       = __check_param(sector,       'sector')
    region       = __check_param(region,       'region', SEARCH_REGION)
    country_code = __check_param(country_code, 'country code')
    partner      = __check_param(partner,      'partner')
    q            = __check_param(q,            'search string', single=True)
    sort_by      = __check_param(sort_by,      'sort_by', SEARCH_SORT, True)

    qopts = {'status':status, 'gender':gender, 'sector':sector,
             'region':region, 'country_code':country_code,
             'partner':partner, 'q':q, 'sort_by': sort_by,
             'page': page}
    for k in filter(lambda x: qopts[x]=='', qopts):
        del qopts[k]
    url = 'loans/search.json?' + urlencode(qopts)
    return __make_call(url, 'loans', searchLoans, opts)
    

def __check_param(value, name, allowed=None, single=False):
    if not value:
        return ''
    bogus = None
    if type(value) == type(''):
        if  value.lower() not in allowed:
            print "%s not in %s" % (value.lower(), str(allowed))
            bogus = [value]
    else:
        if single:
            raise("%s must be a single value, not a list" % name)
        if allowed:
            bogus = filter(lambda x: x.lower() not in allowed, value)
        value = ','.join(value)

    if bogus:
        print type(value)
        raise("Invalid %s: %s. Must be one of %s" %
              (name, ", ".join(bogus), ", ".join(allowed)))
    return value

def __make_call(url, key=None, method=None, args=[]):
    u = urllib2.urlopen(urljoin(BASE_URL, url))
    raw = json.load(u)
    u.close()

    data = key and raw[key] or raw
        
    obj = None
    if type(data) == ListType:
        obj = KivaList()
        for tmp in data:
            spam = KivaContainer(tmp)
            obj.append(spam)
    else:
        obj = KivaContainer(data)
        
    if raw.has_key('paging'):
        current = raw['paging']['page']
        total = raw['paging']['pages']
        obj.current_page = current
        obj.total_pages = total
        obj.page_size = raw['paging']['page_size']
        obj.total_count = raw['paging']['total']
        obj.next_page = current < total and current +1 or None
        obj.prev_page = current > 1 and current - 1 or None

        if method:
            if obj.next_page:
                if type(args) == ListType:
                    qargs = args+[obj.next_page]
                    obj.getNextPage = lambda: method(*qargs)
                else:
                    args['page'] = obj.next_page
                    obj.getNextPage = lambda: method(**args)
            else:
                obj.getNextPage = lambda: None

            if obj.prev_page:
                if type(args) == ListType:
                    qargs = args+[obj.prev_page]
                    obj.getPreviousPage = lambda: method(*qargs)
                else:
                    args['page'] = obj.prev_page
                    obj.getPreviousPage = lambda: method(**args)

            else:
                obj.getPreviousPage = lambda: None
    
    return obj

class KivaContainer(object):
    def __init__(self, data=None):
        if data:
            self.parse(data)

    def parse(self, data):
        for key in data.keys():
            value = data[key]
            param = None
            if type(value) == DictType:
                param = KivaContainer(value)
            elif RE_DATE.match(key):
                param = datetime.strptime(value, FORMAT)
            else:
                param = value
            self.__setattr__(key, param)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def keys(self):
        return self.__dict__.keys()

class KivaList(list, object):
    pass
