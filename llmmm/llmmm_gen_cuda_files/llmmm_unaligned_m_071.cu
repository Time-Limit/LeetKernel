#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_71_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<71> mm_instantiator;

public:
  UnalignedM_71_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_71_MMInstantiatorWrapper__;

}  // namespace LLMMM
